"""可切換的 ASR 後端。

中英夾雜（code-switching）是本專案最大的技術風險，而它沒辦法靠查文件解決 ——
只能同一段話餵給不同模型，直接看誰的輸出可用。所以這裡把模型做成註冊表，
UI 上可以隨時換，換完下一句就生效。

whisper 系列的已知弱點是句內混說時會整句翻成單一語言（例如把英文詞硬翻成中文，
或反過來把中文音譯成英文）。initial_prompt 給一段中英夾雜的範例文字可以明顯改善，
所以 prompt 也開成可調參數而不是寫死。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from importlib.util import find_spec

import numpy as np

SR = 16_000

# 預設 prompt。whisper 會模仿 prompt 的文體，給它中英夾雜的範例
# 能降低「整句被翻成同一種語言」的機率。
DEFAULT_PROMPT = "以下是一段中英文夾雜的會議對話。例如：這個 feature 的 scope 要再 confirm 一下，我下午 sync 給你。"

# whisper 在貪婪解碼下會掉進重複迴圈（同一句話吐幾十次）。
# 官方的解法是溫度回退：用 compression_ratio 偵測到重複就升溫重試。
# temperature 寫死 0.0 會關掉這個機制，實測會產生整段垃圾輸出。
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD = 2.4     # 高於此視為重複，升溫重試
LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6


@dataclass
class AsrResult:
    text: str
    lang: str = ""
    latency_ms: int = 0


@dataclass
class ModelSpec:
    id: str
    label: str
    backend: str                 # 'mlx' | 'faster-whisper' | 'funasr'
    repo: str
    short: str = ""                     # UI 上的短名（不含後端），空的就用 label
    engine: str = ""                    # 後端標籤，用來區分同一個模型的不同執行方式
    # funasr 專用：從哪個 hub 下載。預設 ModelScope（"ms"）在台灣實測只有
    # 70 kB/s，936MB 的 SenseVoice 要跑 3.5 小時；HuggingFace 連線快 20 倍以上。
    hub: str = ""
    note: str = ""
    requires: tuple[str, ...] = ()      # 需要的 import 名稱
    pip: tuple[str, ...] = ()           # 對應的 pip 套件名（不一定同名）
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def installed(self) -> bool:
        return all(find_spec(p) is not None for p in self.requires)

    @property
    def missing(self) -> list[str]:
        """還缺哪些 import。用來判斷安裝完成了沒有。"""
        return [p for p in self.requires if find_spec(p) is None]

    @property
    def pip_packages(self) -> tuple[str, ...]:
        return self.pip or self.requires


# 依「先試哪個」的順序排。turbo 是速度與品質的平衡點，適合當即時的預設；
# large-v3 非 turbo 品質較好但慢；SenseVoice 對中文最強，值得比對。
MODELS: list[ModelSpec] = [
    ModelSpec(
        id="mlx-turbo", label="Whisper large-v3-turbo (MLX / GPU)", backend="mlx",
        repo="mlx-community/whisper-large-v3-turbo",
        short="large-v3-turbo", engine="MLX",
        note="Apple Silicon GPU 加速，即時的預設選擇。混說品質待實測。",
        requires=("mlx_whisper",), pip=("mlx-whisper",), tags=("recommended", "realtime")),
    ModelSpec(
        id="mlx-large-v3", label="Whisper large-v3 (MLX / GPU)", backend="mlx",
        repo="mlx-community/whisper-large-v3-mlx",
        short="large-v3", engine="MLX",
        note="比 turbo 準但慢約 2-3 倍，適合拿來判斷 turbo 到底差多少。",
        requires=("mlx_whisper",), pip=("mlx-whisper",), tags=("quality",)),
    ModelSpec(
        id="mlx-medium", label="Whisper medium (MLX / GPU)", backend="mlx",
        repo="mlx-community/whisper-medium-mlx",
        short="medium", engine="MLX",
        note="體積小、延遲低，混說時通常比 large 明顯差。",
        requires=("mlx_whisper",), pip=("mlx-whisper",), tags=("fast",)),
    ModelSpec(
        id="mlx-small", label="Whisper small (MLX / GPU)", backend="mlx",
        repo="mlx-community/whisper-small-mlx",
        short="small", engine="MLX",
        note="最快。當作延遲的下界參考。",
        requires=("mlx_whisper",), pip=("mlx-whisper",), tags=("fast",)),
    ModelSpec(
        id="fw-turbo", label="Whisper large-v3-turbo (faster-whisper / CPU)",
        backend="faster-whisper", repo="large-v3-turbo",
        short="large-v3-turbo", engine="CPU",
        note="ctranslate2 在 macOS 只有 CPU，慢於 MLX。留著是為了跨平台對照。",
        requires=("faster_whisper",), pip=("faster-whisper",), tags=("cpu",)),
    ModelSpec(
        id="fw-medium", label="Whisper medium (faster-whisper / CPU)",
        backend="faster-whisper", repo="medium",
        short="medium", engine="CPU", note="純 CPU 對照組。",
        requires=("faster_whisper",), pip=("faster-whisper",), tags=("cpu",)),
    # 這兩個走 HuggingFace 而非 funasr 預設的 ModelScope。
    # 實測 ModelScope 在台灣只有 70 kB/s，936MB 的 SenseVoice 要下載 3.5 小時；
    # HuggingFace 的連線時間快 20 倍以上。repo id 直接用 HF 上的，
    # 不要用 funasr 的別名對照表 —— SenseVoice 在那張表裡沒有 HF 對應。
    ModelSpec(
        id="sensevoice", label="SenseVoice Small (FunASR)", backend="funasr",
        repo="FunAudioLLM/SenseVoiceSmall", hub="hf",
        short="SenseVoice Small", engine="FunASR",
        note="非自回歸、對中文最強，官方支援中英日韓粵。不逐字續寫，結構上不會掉進"
             "重複迴圈，但通常較不流暢。混說表現值得跟 whisper 直接對打。約 936MB。",
        requires=("funasr",), pip=("funasr", "modelscope"), tags=("chinese",)),
    ModelSpec(
        id="paraformer-stream", label="Paraformer 串流 (FunASR)", backend="funasr",
        repo="funasr/paraformer-zh-streaming", hub="hf",
        short="Paraformer 串流", engine="FunASR",
        note="唯一的真串流模型，延遲最低，但以中文為主，英文詞很可能被吃掉。",
        requires=("funasr",), pip=("funasr", "modelscope"), tags=("chinese", "streaming")),
]

MODELS_BY_ID = {m.id: m for m in MODELS}
DEFAULT_MODEL_ID = "mlx-turbo"


def strip_repetition(text: str, max_repeats: int = 3) -> str:
    """砍掉重複迴圈。

    whisper 內建的溫度回退（靠 compression_ratio 偵測）並非每次都救得回來，
    實測在雜訊或非語音片段上仍會吐出同一個詞數百次。即時工具不能讓這種輸出
    直接落進逐字稿，所以再加一道後處理：偵測尾端的 n-gram 迴圈就截斷。
    """
    if not text:
        return text
    # 兩種切法：以空白分隔的詞（英文為主）、逐字（中文沒有空白）
    for unit, sep in ((text.split(), " "), (list(text), "")):
        n = len(unit)
        if n < max_repeats * 2:
            continue
        for size in range(1, 13):
            if n < size * (max_repeats + 1):
                break
            tail = unit[-size:]
            reps = 1
            while reps * size < n and unit[n - (reps + 1) * size: n - reps * size] == tail:
                reps += 1
            if reps > max_repeats:
                # 保留 max_repeats 次就好，後面全是迴圈
                keep = unit[: n - (reps - max_repeats) * size]
                return sep.join(keep).strip()
    return text


class AsrBackend:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def load(self) -> None: ...

    def transcribe(self, audio: np.ndarray, prompt: str = "",
                   language: str | None = None) -> AsrResult:
        raise NotImplementedError


class MlxWhisper(AsrBackend):
    def load(self) -> None:
        import mlx_whisper

        # 用一秒靜音暖機：權重下載與載入都發生在這裡，
        # 不要讓使用者的第一句話扛掉幾十秒的載入時間。
        mlx_whisper.transcribe(np.zeros(SR, dtype=np.float32),
                               path_or_hf_repo=self.spec.repo, verbose=None)

    def transcribe(self, audio, prompt="", language=None) -> AsrResult:
        import mlx_whisper

        t0 = time.perf_counter()
        r = mlx_whisper.transcribe(
            audio.astype(np.float32), path_or_hf_repo=self.spec.repo,
            language=language, initial_prompt=prompt or None,
            condition_on_previous_text=False,   # 避免前一句的錯誤滾雪球
            temperature=TEMPERATURE_FALLBACK,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            fp16=True, verbose=None,
        )
        return AsrResult(text=strip_repetition((r.get("text") or "").strip()),
                         lang=r.get("language", "") or "",
                         latency_ms=int((time.perf_counter() - t0) * 1000))


class FasterWhisper(AsrBackend):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._m = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        # macOS 上 ctranslate2 沒有 Metal 後端，只能 CPU；int8 是可用的折衷。
        self._m = WhisperModel(self.spec.repo, device="cpu", compute_type="int8")
        list(self._m.transcribe(np.zeros(SR, dtype=np.float32))[0])   # 暖機

    def transcribe(self, audio, prompt="", language=None) -> AsrResult:
        if self._m is None:
            self.load()
        t0 = time.perf_counter()
        segs, info = self._m.transcribe(
            audio.astype(np.float32), language=language,
            initial_prompt=prompt or None, condition_on_previous_text=False,
            temperature=TEMPERATURE_FALLBACK,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=LOGPROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            beam_size=1, vad_filter=False,   # VAD 已經在上游做過了
        )
        text = strip_repetition("".join(s.text for s in segs).strip())
        return AsrResult(text=text, lang=getattr(info, "language", "") or "",
                         latency_ms=int((time.perf_counter() - t0) * 1000))


class FunAsr(AsrBackend):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._m = None
        self._streaming = "streaming" in spec.tags
        self._cache: dict = {}

    def load(self) -> None:
        from funasr import AutoModel

        kw = {"model": self.spec.repo, "disable_update": True, "device": "cpu"}
        if self.spec.hub:
            kw["hub"] = self.spec.hub
        if not self._streaming:
            kw["vad_model"] = None
        self._m = AutoModel(**kw)

    def transcribe(self, audio, prompt="", language=None) -> AsrResult:
        if self._m is None:
            self.load()
        t0 = time.perf_counter()
        kw: dict = {}
        if self._streaming:
            # Paraformer online 需要 chunk 設定；這裡當成一次性 offline 呼叫，
            # 真要用它的串流能力得改成持續餵 cache，屆時再處理。
            kw = {"chunk_size": [0, 10, 5], "encoder_chunk_look_back": 4,
                  "decoder_chunk_look_back": 1, "is_final": True, "cache": self._cache}
        else:
            kw = {"language": "auto", "use_itn": True}
        res = self._m.generate(input=audio.astype(np.float32), **kw)
        text = ""
        if isinstance(res, list) and res:
            text = str(res[0].get("text", ""))
        # SenseVoice 會夾帶 <|zh|><|NEUTRAL|> 之類的標記，去掉
        while "<|" in text and "|>" in text:
            a, b = text.index("<|"), text.index("|>")
            if b < a:
                break
            text = text[:a] + text[b + 2:]
        return AsrResult(text=strip_repetition(text.strip()), lang="",
                         latency_ms=int((time.perf_counter() - t0) * 1000))


_BACKENDS = {"mlx": MlxWhisper, "faster-whisper": FasterWhisper, "funasr": FunAsr}
_CACHE: dict[str, AsrBackend] = {}
_CACHE_LOCK = threading.Lock()
_LOAD_LOCKS: dict[str, threading.Lock] = {}


def get_backend(model_id: str) -> AsrBackend:
    """取得模型後端，第一次呼叫會下載權重並暖機。

    每個模型一把載入鎖。少了它，兩條執行緒同時要同一個還沒下載的模型時
    會各自開始下載 —— 症狀是「Still waiting to acquire lock on ...」
    加上兩份平行的下載互搶頻寬。會發生在使用者重按「開始錄音」，
    或設定變更與錄音啟動同時觸發載入的時候。
    """
    spec = MODELS_BY_ID.get(model_id)
    if spec is None:
        raise ValueError(f"未知的模型 id: {model_id}")
    if not spec.installed:
        missing = [p for p in spec.requires if find_spec(p) is None]
        raise RuntimeError(f"{spec.label} 需要先安裝：pip install {' '.join(missing)}")

    hit = _CACHE.get(model_id)
    if hit is not None:
        return hit
    with _CACHE_LOCK:
        lock = _LOAD_LOCKS.setdefault(model_id, threading.Lock())
    with lock:
        hit = _CACHE.get(model_id)      # 等鎖的期間別人可能已經載好了
        if hit is not None:
            return hit
        b = _BACKENDS[spec.backend](spec)
        b.load()
        _CACHE[model_id] = b
        return b


def catalog() -> list[dict]:
    return [{"id": m.id, "label": m.label, "short": m.short or m.label,
             "engine": m.engine, "backend": m.backend, "note": m.note,
             "installed": m.installed, "tags": list(m.tags),
             "missing": m.missing, "pip": list(m.pip_packages),
             "loaded": m.id in _CACHE} for m in MODELS]
