"""麥克風擷取。輸出固定為 16 kHz / mono / float32，下游模型都吃這個格式。

系統音訊（線上會議對方的聲音）不在這裡 —— 那是分平台的工作，且 CLAUDE.md 建議
與麥克風分軌處理，不要混成單軌。這個模組刻意只管一條軌，多軌是上層的事。
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

TARGET_SR = 16_000


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_sr: int
    is_default: bool
    is_loopback: bool = False


# 虛擬音訊裝置的常見名稱。裝了這類裝置之後，「系統在播什麼」就會變成一個
# 普通的輸入裝置 —— YouTube、Google Meet、任何 App 的聲音都能直接收，
# 不需要 ScreenCaptureKit 也不需要任何原生程式碼。
LOOPBACK_HINTS = ("blackhole", "loopback", "soundflower", "virtual", "aggregate",
                  "multi-output", "vb-audio", "voicemeeter", "monitor of ", "stereo mix",
                  "what-u-hear", "wasapi", "pulse", "pipewire")


def looks_like_loopback(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in LOOPBACK_HINTS)


def list_input_devices() -> list[DeviceInfo]:
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append(DeviceInfo(
                index=i, name=d["name"], channels=int(d["max_input_channels"]),
                default_sr=int(d["default_samplerate"]), is_default=(i == default_in),
                is_loopback=looks_like_loopback(d["name"]),
            ))
    return out


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    n = int(round(len(x) * dst_sr / src_sr))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    # 線性插值。VAD/ASR 前端都會再過 mel filterbank，這裡的品質差異可忽略，
    # 換來不必把 torchaudio 拉進音訊執行緒。
    src_t = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float64)
    return np.interp(dst_t, src_t, x).astype(np.float32)


# 刻意不做自動增益：實測在只有底噪的音流上放大 24 倍，VAD 就會誤判成語音，
# whisper 接著生成幻覺文字。要做的話必須用「調變深度」當閘門（語音在 100ms
# 尺度的動態範圍 >20 dB、穩態噪音 <8 dB），單看振幅一定會誤觸發。


class MicSource:
    """背景執行緒收音，主流程用 read() 拉 16k mono float32。

    佇列滿了就丟最舊的一塊：即時逐字稿寧可掉音訊也不要愈跑愈延遲。
    掉塊會記在 dropped，UI 可以顯示出來，不要讓它靜默發生。
    """

    def __init__(self, device: int | None = None, blocksize: int = 1024,
                 max_queue_blocks: int = 200):
        self.device = device
        self.blocksize = blocksize
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_blocks)
        self._stream: sd.InputStream | None = None
        self._src_sr = TARGET_SR
        self._lock = threading.Lock()
        self.dropped = 0
        self.level = 0.0          # 給 UI 畫音量條用的 RMS

    @property
    def source_samplerate(self) -> int:
        return self._src_sr

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        mono = indata[:, 0] if indata.ndim > 1 else indata
        chunk = _resample(np.asarray(mono, dtype=np.float32), self._src_sr, TARGET_SR)
        if chunk.size:
            self.level = float(np.sqrt(np.mean(chunk ** 2)))
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            with self._lock:
                self.dropped += 1
            try:
                self._q.get_nowait()
                self._q.put_nowait(chunk)
            except queue.Empty:
                pass

    def start(self) -> None:
        if self._stream is not None:
            return
        # 先試著直接開 16k，多數 macOS 裝置可以由 CoreAudio 代為轉換；
        # 不行就用裝置原生取樣率收，自己降頻。
        for sr in (TARGET_SR, None):
            try:
                self._src_sr = sr or int(
                    sd.query_devices(self.device if self.device is not None else
                                     sd.default.device[0])["default_samplerate"])
                self._stream = sd.InputStream(
                    device=self.device, channels=1, samplerate=self._src_sr,
                    dtype="float32", blocksize=self.blocksize, callback=self._callback,
                )
                self._stream.start()
                return
            except Exception:
                self._stream = None
                continue
        raise RuntimeError("無法開啟麥克風。請確認系統設定 → 隱私權與安全性 → 麥克風已授權。")

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
