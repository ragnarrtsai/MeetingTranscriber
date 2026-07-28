"""Silero VAD 語句切分。

whisper 不是真串流（CLAUDE.md 風險 #2）。這裡的處理方式是：
用 VAD 找語句邊界，語句結束就送去做「定稿」轉寫；語句還在進行中的話，
每隔一段時間把「目前累積的音訊」重跑一次當「暫定稿」。

這樣延遲來自 partial_interval 而非模型本身，而定稿永遠是在完整語句上做的，
不會有固定切窗把字切一半的問題。代價是同一段音訊會被轉寫多次。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

FRAME = 512               # silero 在 16 kHz 下要求的固定窗長
SR = 16_000
FRAME_MS = FRAME * 1000 // SR   # 32 ms


class EventKind(Enum):
    PARTIAL = "partial"       # 語句進行中的暫定稿
    FINAL = "final"           # 語句結束的定稿
    LEVEL = "level"


@dataclass
class Utterance:
    id: int
    start_ms: int
    end_ms: int
    audio: np.ndarray
    is_final: bool
    # 錄音模式下音訊會先落到磁碟，audio 留空、由 path 指向暫存檔。
    # 這樣 ASR 佇列裡只放路徑，VAD 永遠不會被 ASR 反壓擋住，
    # 前面兩層音訊佇列也就不會滿到必須丟資料。
    path: str = ""


@dataclass
class VadConfig:
    threshold: float = 0.5
    min_speech_ms: int = 250          # 太短的當雜音丟掉
    min_silence_ms: int = 600         # 靜音多久算語句結束（也是定稿延遲的下限）
    speech_pad_ms: int = 200          # 前後各留一點，避免咬掉字頭字尾
    max_utterance_ms: int = 20_000    # 有人一直講不停就強制斷句
    partial_every_ms: int = 1_200     # 暫定稿頻率
    # 音訊太短時 whisper 會硬湊出字（實測 0.5 秒可以幻覺出「AAA」），
    # 所以暫定稿的門檻比語句門檻高，寧可晚一點才出現第一行字。
    min_partial_ms: int = 900
    min_embed_ms: int = 700           # 短於此的語句不做聲紋（向量不可靠）


@dataclass
class _State:
    in_speech: bool = False
    speech_frames: int = 0
    silence_frames: int = 0
    buf: list[np.ndarray] = field(default_factory=list)
    buf_samples: int = 0
    start_ms: int = 0
    last_partial_ms: int = 0


class VadSegmenter:
    def __init__(self, cfg: VadConfig | None = None):
        from silero_vad import load_silero_vad

        self.cfg = cfg or VadConfig()
        self._model = load_silero_vad()
        self._torch = __import__("torch")
        self._st = _State()
        self._carry = np.zeros(0, dtype=np.float32)
        self._pad = np.zeros(0, dtype=np.float32)   # 上一段的尾巴，當下一段的前置 padding
        self._elapsed_ms = 0
        self._utt_id = 0

    def reset(self) -> None:
        self._model.reset_states()
        self._st = _State()
        self._carry = np.zeros(0, dtype=np.float32)
        self._pad = np.zeros(0, dtype=np.float32)
        self._elapsed_ms = 0

    def _prob(self, frame: np.ndarray) -> float:
        t = self._torch.from_numpy(frame).unsqueeze(0)
        with self._torch.no_grad():
            return float(self._model(t, SR).item())

    def push(self, chunk: np.ndarray) -> list[Utterance]:
        """餵一塊音訊，回傳這塊產生的 partial / final 語句（通常是 0 或 1 個）。"""
        cfg = self.cfg
        out: list[Utterance] = []
        buf = np.concatenate([self._carry, chunk]) if self._carry.size else chunk
        n_frames = len(buf) // FRAME
        self._carry = buf[n_frames * FRAME:].copy()

        pad_frames = max(1, cfg.speech_pad_ms // FRAME_MS)
        min_speech_f = max(1, cfg.min_speech_ms // FRAME_MS)
        min_sil_f = max(1, cfg.min_silence_ms // FRAME_MS)
        st = self._st

        for i in range(n_frames):
            frame = buf[i * FRAME:(i + 1) * FRAME]
            self._elapsed_ms += FRAME_MS
            p = self._prob(frame)

            if not st.in_speech:
                # 未進入語音：只留最近幾個 frame 當 padding，其餘丟掉
                self._pad = np.concatenate([self._pad, frame])[-pad_frames * FRAME:]
                if p >= cfg.threshold:
                    st.in_speech = True
                    st.speech_frames = 1
                    st.silence_frames = 0
                    st.buf = [self._pad.copy(), frame]
                    st.buf_samples = self._pad.size + FRAME
                    st.start_ms = max(0, self._elapsed_ms - self._pad.size * 1000 // SR)
                    st.last_partial_ms = self._elapsed_ms
                continue

            st.buf.append(frame)
            st.buf_samples += FRAME
            if p >= cfg.threshold:
                st.speech_frames += 1
                st.silence_frames = 0
            else:
                st.silence_frames += 1

            dur_ms = st.buf_samples * 1000 // SR
            ended = st.silence_frames >= min_sil_f
            too_long = dur_ms >= cfg.max_utterance_ms

            if ended or too_long:
                if st.speech_frames >= min_speech_f:
                    self._utt_id += 1
                    audio = np.concatenate(st.buf)
                    out.append(Utterance(id=self._utt_id, start_ms=st.start_ms,
                                         end_ms=self._elapsed_ms, audio=audio, is_final=True))
                # 強制斷句時保留尾巴接續，正常結束就清空
                self._pad = np.concatenate(st.buf)[-pad_frames * FRAME:] if too_long \
                    else np.zeros(0, dtype=np.float32)
                self._st = st = _State()
                if too_long:
                    self._model.reset_states()
                continue

            if self._elapsed_ms - st.last_partial_ms >= cfg.partial_every_ms \
                    and dur_ms >= cfg.min_partial_ms:
                st.last_partial_ms = self._elapsed_ms
                out.append(Utterance(id=self._utt_id + 1, start_ms=st.start_ms,
                                     end_ms=self._elapsed_ms,
                                     audio=np.concatenate(st.buf), is_final=False))
        return out

    def flush(self) -> Utterance | None:
        """停止錄音時把手上未結束的語句吐出來，不要讓最後一句消失。"""
        st = self._st
        if st.in_speech and st.speech_frames >= max(1, self.cfg.min_speech_ms // FRAME_MS):
            self._utt_id += 1
            u = Utterance(id=self._utt_id, start_ms=st.start_ms, end_ms=self._elapsed_ms,
                          audio=np.concatenate(st.buf), is_final=True)
            self._st = _State()
            return u
        return None
