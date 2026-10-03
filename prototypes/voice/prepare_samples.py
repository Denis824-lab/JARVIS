"""Generate personal, local Russian voice auditions with explicit model sources."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import platform
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models" / "voice-probe"
OUTPUT = ROOT / "assets" / "voice-samples"
RATE = 24000
TEXTS = [
    "На связи. Что нужно сделать?",
    "Открываю браузер на ноутбуке. Готово.",
    "Завтра, четвёртого октября, в девятнадцать часов нужно проверить проект.",
    "Доход за выбранный период — двенадцать тысяч четыреста восемьдесят рублей. Это на восемь процентов больше прошлого периода.",
    "Пришёл ответ на твоё письмо. Прочитать главное или открыть переписку?",
    "Соединение прервалось. Файл уже сохранён, а сообщение ещё не отправлено.",
    "Я нашёл два варианта. Первый проще в поддержке, второй быстрее на больших объёмах. Давай сравним их на твоей задаче.",
]
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
SILERO_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "JARVIS-voice-probe/1.0"})
            with urllib.request.urlopen(request, timeout=40) as response, temporary.open("wb") as out:
                expected = response.headers.get("Content-Length")
                received = 0
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    received += len(chunk)
            if received == 0 or (expected and received != int(expected)):
                raise OSError("Incomplete model download")
            temporary.replace(destination)
            logging.info("Downloaded %s (%.1f MB)", destination.name, received / 1e6)
            return destination
        except (OSError, TimeoutError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f"Could not download {destination.name}: {error}") from error
            time.sleep(attempt + 1)
    raise RuntimeError("Download did not complete")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_sample(label: str, utterances: list[tuple[np.ndarray, int]]) -> dict:
    resampled = []
    for samples, rate in utterances:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size < rate / 10 or not np.isfinite(audio).all():
            raise ValueError("Invalid or empty generated audio")
        if rate != RATE:
            count = round(audio.size * RATE / rate)
            audio = np.interp(np.arange(count) * rate / RATE, np.arange(audio.size), audio).astype(np.float32)
        resampled.extend([audio, np.zeros(round(RATE * 0.4), dtype=np.float32)])
    combined = np.concatenate(resampled[:-1])
    rms = float(np.sqrt(np.mean(combined.astype(np.float64) ** 2)))
    if rms < 1e-5:
        raise ValueError("Generated sample is silent")
    gain = min(0.1 / rms, 0.95 / float(np.max(np.abs(combined))))
    pcm = np.round(np.clip(combined * gain, -1, 1) * 32767).astype("<i2")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / f"voice-{label}.wav"
    temporary = destination.with_suffix(".tmp")
    with wave.open(str(temporary), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())
    temporary.replace(destination)
    return {
        "file": destination.name,
        "seconds": round(len(pcm) / RATE, 3),
        "sample_rate": RATE,
        "peak": round(float(np.max(np.abs(pcm.astype(np.float32)))) / 32767, 4),
        "sha256": digest(destination),
        "normalization": "same whole-clip RMS target with peak ceiling; not LUFS matching",
    }


def run_piper(label: str, name: str) -> dict:
    from piper import PiperVoice, SynthesisConfig

    folder = CACHE / name
    file_name = f"ru_RU-{name}-medium.onnx"
    base = f"{PIPER_BASE}/ru/ru_RU/{name}/medium"
    sources = [
        (f"{base}/{file_name}", folder / file_name),
        (f"{base}/{file_name}.json", folder / f"{file_name}.json"),
        (f"{base}/MODEL_CARD", folder / "MODEL_CARD"),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda pair: download(*pair), sources))
    logging.info("Voice %s: loading Piper %s", label, name)
    start = time.perf_counter()
    voice = PiperVoice.load(str(folder / file_name), use_cuda=False)
    load_seconds = time.perf_counter() - start
    config = SynthesisConfig(length_scale=1.0, normalize_audio=False)
    utterances, times = [], []
    for text in TEXTS:
        began = time.perf_counter()
        pieces, first_chunk, rate = [], None, None
        for chunk in voice.synthesize(text, syn_config=config):
            if first_chunk is None:
                first_chunk = time.perf_counter() - began
            rate = chunk.sample_rate
            pieces.append(np.frombuffer(chunk.audio_int16_bytes, dtype="<i2").astype(np.float32) / 32768)
        if not pieces or rate is None:
            raise RuntimeError("Piper produced no audio")
        utterances.append((np.concatenate(pieces), rate))
        times.append({"generation_seconds": time.perf_counter() - began, "first_chunk_seconds": first_chunk})
    result = write_sample(label, utterances)
    result.update({"engine": "Piper", "voice": name, "load_seconds": round(load_seconds, 3), "utterances": times,
                   "rss_after_synthesis_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
                   "model_sha256": digest(folder / file_name), "model_card": (folder / "MODEL_CARD").read_text(encoding="utf-8")})
    del voice, utterances
    gc.collect()
    return result


def run_silero() -> dict[str, dict]:
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    path = download(SILERO_URL, CACHE / "silero" / "v5_5_ru.pt")
    license_path = download("https://raw.githubusercontent.com/snakers4/silero-models/master/LICENSE", CACHE / "silero" / "LICENSE")
    logging.info("Loading official Silero Russian package on CPU")
    began = time.perf_counter()
    model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    load_seconds = time.perf_counter() - began
    results = {}
    for label, speaker in [("C", "aidar"), ("D", "eugene")]:
        if speaker not in model.speakers:
            raise ValueError(f"Speaker {speaker} is unavailable in loaded model")
        logging.info("Voice %s: Silero %s", label, speaker)
        utterances, times = [], []
        with torch.inference_mode():
            for text in TEXTS:
                start = time.perf_counter()
                audio = model.apply_tts(text=text, speaker=speaker, sample_rate=RATE, put_accent=True, put_yo=True)
                elapsed = time.perf_counter() - start
                utterances.append((audio.detach().cpu().numpy(), RATE))
                times.append({"generation_seconds": elapsed, "first_chunk_seconds": None,
                              "note": "full-utterance API; no streaming first-chunk measurement"})
        result = write_sample(label, utterances)
        result.update({"engine": "Silero", "voice": speaker, "model": "v5_5_ru", "load_seconds_shared": round(load_seconds, 3),
                       "threads": 4, "utterances": times, "rss_after_synthesis_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
                       "model_sha256": digest(path), "license_file": str(license_path.relative_to(ROOT))})
        results[label] = result
    del model
    gc.collect()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["piper", "silero", "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report.update({"date": "2026-10-03", "python": platform.python_version(), "platform": platform.platform(),
                   "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1), "mode": "CPU, local synthetic text, no paid API",
                   "texts": TEXTS, "limitations": "initial audition only; no game-load, cancellation or Linux tests; RSS is not isolated peak memory"})
    samples = report.setdefault("samples", {})
    failures = {}
    if args.engine in ("piper", "all"):
        for label, name in [("A", "denis"), ("B", "dmitri")]:
            try:
                samples[label] = run_piper(label, name)
            except Exception as error:
                failures[label] = f"{type(error).__name__}: {error}"
                logging.exception("Voice %s failed", label)
    if args.engine in ("silero", "all"):
        try:
            samples.update(run_silero())
        except Exception as error:
            failures["silero"] = f"{type(error).__name__}: {error}"
            logging.exception("Silero failed")
    report["last_run_failures"] = failures
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved %d samples; %d failures", len(samples), len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
