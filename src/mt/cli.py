"""命令列工具。主要用途是不開 UI 就能驗證模型。

  python run.py serve                                 # 開 UI（預設）
  python run.py models                                # 看有哪些模型可用
  python run.py file 錄音.m4a --model mlx-turbo        # 單一音檔跑一次
  python run.py compare 錄音.m4a --models mlx-turbo,mlx-large-v3
                                                      # 同一段話多個模型並排，判斷混說品質
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import asr as asr_mod
from .vad import SR, VadConfig, VadSegmenter


def load_audio(path: str | Path) -> np.ndarray:
    """任何格式都先用 ffmpeg 轉成 16k mono float32。"""
    path = str(path)
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr == SR:
            return mono.astype(np.float32)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", str(SR), "-f", "wav", out], check=True)
    import soundfile as sf

    data, _ = sf.read(out, dtype="float32")
    Path(out).unlink(missing_ok=True)
    return np.asarray(data, dtype=np.float32).reshape(-1)


def split_utterances(audio: np.ndarray, cfg: VadConfig | None = None) -> list:
    cfg = cfg or VadConfig(partial_every_ms=10**9)   # 離線不需要暫定稿
    seg = VadSegmenter(cfg)
    utts = []
    step = SR  # 一秒一塊餵進去，模擬串流
    for i in range(0, len(audio), step):
        utts += [u for u in seg.push(audio[i:i + step]) if u.is_final]
    tail = seg.flush()
    if tail is not None:
        utts.append(tail)
    return utts


def cmd_models(_: argparse.Namespace) -> int:
    for m in asr_mod.catalog():
        mark = "✓" if m["installed"] else "✗"
        print(f" {mark} {m['id']:<20} {m['label']}")
        print(f"   {' ' * 20} {m['note']}")
    return 0


def cmd_file(a: argparse.Namespace) -> int:
    audio = load_audio(a.path)
    print(f"音檔 {len(audio)/SR:.1f} 秒 → VAD 切分中…", file=sys.stderr)
    utts = split_utterances(audio)
    print(f"切出 {len(utts)} 句\n", file=sys.stderr)
    backend = asr_mod.get_backend(a.model)
    total = 0
    for u in utts:
        r = backend.transcribe(u.audio, prompt="" if a.no_prompt else asr_mod.DEFAULT_PROMPT,
                               language=a.language)
        total += r.latency_ms
        ts = f"{u.start_ms // 60000:02d}:{u.start_ms // 1000 % 60:02d}"
        print(f"[{ts}] {r.text}", flush=True)
    audio_ms = int(len(audio) / SR * 1000)
    print(f"\n共 {total} ms 推論 / {audio_ms} ms 音訊 → RTF {total/max(1,audio_ms):.2f}"
          f"（<1 才有機會即時）")
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    """同一段音訊餵給多個模型並排輸出。判斷中英夾雜品質最直接的方式。"""
    audio = load_audio(a.path)
    utts = split_utterances(audio)
    ids = [m.strip() for m in a.models.split(",") if m.strip()]
    prompt = "" if a.no_prompt else asr_mod.DEFAULT_PROMPT
    backends = {}
    for i in ids:
        try:
            backends[i] = asr_mod.get_backend(i)
        except Exception as e:
            print(f"! 跳過 {i}：{e}", file=sys.stderr)
    if not backends:
        return 1
    rtf = {i: 0 for i in backends}
    for u in utts:
        ts = f"{u.start_ms // 60000:02d}:{u.start_ms // 1000 % 60:02d}"
        print(f"\n[{ts}] ({(u.end_ms - u.start_ms) / 1000:.1f}s)", flush=True)
        for i, b in backends.items():
            r = b.transcribe(u.audio, prompt=prompt, language=a.language)
            rtf[i] += r.latency_ms
            print(f"  {i:<18} {r.text}", flush=True)
    audio_ms = max(1, int(len(audio) / SR * 1000))
    print("\nRTF（越小越快，<1 才有機會即時）：")
    for i, ms in rtf.items():
        print(f"  {i:<18} {ms / audio_ms:.2f}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    print(f"  meeting-transcriber → http://{a.host}:{a.port}")
    uvicorn.run(create_app(a.db), host=a.host, port=a.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="meeting-transcriber")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="開啟 UI（預設）")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8823)
    s.add_argument("--db", default="data/meetings.db")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("models", help="列出可用模型")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("file", help="轉寫單一音檔")
    s.add_argument("path")
    s.add_argument("--model", default=asr_mod.DEFAULT_MODEL_ID)
    s.add_argument("--language", default=None)
    s.add_argument("--no-prompt", action="store_true", help="不給混說提示詞，看差多少")
    s.set_defaults(fn=cmd_file)

    s = sub.add_parser("compare", help="多模型並排比對同一段音訊")
    s.add_argument("path")
    s.add_argument("--models", default="mlx-turbo,mlx-large-v3")
    s.add_argument("--language", default=None)
    s.add_argument("--no-prompt", action="store_true")
    s.set_defaults(fn=cmd_compare)

    a = ap.parse_args(argv)
    if not getattr(a, "cmd", None):
        a = ap.parse_args(["serve"])
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
