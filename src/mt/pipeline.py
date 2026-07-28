"""會議流程：多軌音訊 → VAD → ASR ＋ 聲紋 → 分群 → 落地 → 事件。

執行緒配置：

    麥克風 (sounddevice callback) ─→ mic 軌佇列 ─┐
    瀏覽器分頁音訊 (WebSocket)   ─→ system 軌佇列 ┤
                                                 ├→ 每軌一條 VAD 執行緒
                                                 ↓
                                     共用的 ASR 工作執行緒 ─→ 事件佇列 ─→ WebSocket

為什麼要分軌（CLAUDE.md 的判斷，實作後仍然成立）：
系統音訊裡是遠端參與者、麥克風裡是現場的人，分軌讓「現場 vs 遠端」免費切乾淨，
分群只需在各軌內部做。混成單軌反而會因兩個來源的通道特性差異，
把同一個人分成兩個群。

為什麼 ASR 只有一條工作執行緒：
GPU 推論本來就是序列化的，多開執行緒只會互相搶資源並讓延遲更難預測。
反之 VAD 必須每軌獨立且與 ASR 分離 —— ASR 一句要數百毫秒，
擠在一起會讓語句邊界判斷跟著延遲，愈跑愈歪。
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from . import asr as asr_mod
from .audio import MicSource, list_input_devices, modulation_depth_db
from .spk import EMBED_DIM, EMBED_TAG, Embedder, OnlineClusterer
from .store import Store
from .vad import SR, Utterance, VadConfig, VadSegmenter

MIC = "mic"
SYSTEM = "system"

# 記憶體模式的待處理上限。超過就讓 VAD 執行緒等，形成反壓 ——
# 代價是上游的音訊佇列最終會滿而丟資料。錄音模式不受這個限制。
MEM_PENDING = 32
_EMPTY = np.zeros(0, dtype=np.float32)

# 每一軌的說話者代號前綴，避免兩軌撞號。
TRACK_PREFIX = {MIC: "", SYSTEM: "R"}
TRACK_LABEL = {MIC: "現場", SYSTEM: "遠端"}


@dataclass
class SessionConfig:
    # 可以同時選多個模型。第一個是主要的（逐字稿的正文用它、暫定稿也只用它），
    # 其餘每一句都會再跑一次並排顯示 —— 判斷中英夾雜品質最直接的方式。
    # 代價是推論時間隨模型數線性增加，選太多會讓即時性掉下來。
    model_ids: list[str] = field(default_factory=lambda: [asr_mod.DEFAULT_MODEL_ID])
    prompt: str = asr_mod.DEFAULT_PROMPT
    language: str | None = None         # None = 自動偵測；混說時通常讓它自動更好
    device_index: int | None = None
    enable_mic: bool = True
    embed_device: str = "cpu"
    cluster_threshold: float = 0.45
    person_threshold: float = 0.70
    recluster_every: int = 6            # 每累積幾句就全域重算一次（追溯修正）
    enable_diarization: bool = True
    # 錄音模式（預設開）：定稿的音訊先寫到磁碟暫存，轉寫完立刻刪除。
    #
    # 目的不是保留音檔，而是在 ASR 跟不上時不必丟音訊 —— 逐字稿會晚出現但完整。
    # 一句 3 秒的語句約 192KB、一次循序寫入約 1-2ms，相對於 ASR 的上百毫秒是雜訊，
    # 所以沒有理由為了省這點時間去冒丟音訊的風險。
    #
    # 關掉的理由是隱私而不是速度：CLAUDE.md 定案「不留音檔」，暫存檔終究是音訊
    # 落到磁碟，當機時會留下痕跡。關掉就完全走記憶體，代價是過載時會丟音訊。
    # 暫定稿永遠不落地 —— 它一秒多產生一次、落後時本來就該丟。
    spool_audio: bool = True
    spool_max_pending: int = 4000        # 落後這麼多句就連磁碟也不收了（防無上限成長）
    # 每軌音訊佇列的深度（單位：1024 樣本的塊，400 塊約 25.6 秒）。
    # 滿了就開始丟音訊，是「遺漏」的來源。開成可設定主要是為了測試能逼出過載。
    track_queue_blocks: int = 400
    # 語音閘門：段落的調變深度（見 audio.modulation_depth_db）低於這個值就不送進
    # ASR，用來擋非語音被 whisper 轉成幻覺文字。預設 0（關閉），也沒有 UI 入口 ——
    # 幻覺的病因是麥克風收到壞資料，那個修掉之後就用不到它了。留著是因為每段的
    # 實測值會記進 stats.recent_db，幻覺再出現時那些數字就是判斷門檻的依據。
    speech_gate_db: float = 0.0
    vad: VadConfig = field(default_factory=VadConfig)

    @property
    def model_id(self) -> str:
        """主要模型。逐字稿正文與暫定稿都用它。"""
        return self.model_ids[0] if self.model_ids else asr_mod.DEFAULT_MODEL_ID

    @property
    def compare_ids(self) -> list[str]:
        """只做並排比對的模型。"""
        return [m for m in self.model_ids[1:] if m != self.model_id]


# 可以寫的欄位。model_id / compare_ids 是唯讀 property，setattr 會炸。
SETTABLE_FIELDS = frozenset(f.name for f in fields(SessionConfig))


@dataclass
class Track:
    """一條音訊來源。VAD 與分群各自獨立，ASR 與人物庫共用。"""

    name: str
    q: queue.Queue
    seg: VadSegmenter
    clusterer: OnlineClusterer
    # 每個 VadSegmenter 只數自己看過多少音訊，所以時間戳是「從這一軌開始算」。
    # 分頁音訊通常是會議中途才加進來的，不補上偏移，兩軌的逐字稿順序會錯亂。
    offset_ms: int = 0
    thread: threading.Thread | None = None
    seg_ids: list[int] = field(default_factory=list)
    seg_keys: list[str] = field(default_factory=list)
    since_recluster: int = 0
    level: float = 0.0
    dropped: int = 0
    running: bool = True

    @property
    def label(self) -> str:
        return TRACK_LABEL.get(self.name, self.name)


class Session:
    def __init__(self, store: Store, cfg: SessionConfig | None = None):
        self.store = store
        self.cfg = cfg or SessionConfig()
        self.events: queue.Queue[dict] = queue.Queue(maxsize=4000)
        self.meeting_id: int | None = None
        self.running = False

        self._mic: MicSource | None = None
        self._mic_thread: threading.Thread | None = None
        self._embedder = Embedder(device=self.cfg.embed_device)
        self._tracks: dict[str, Track] = {}
        # 佇列開得很深，實際上限由 _submit 依模式決定（記憶體模式 MEM_PENDING）。
        self._jobs: queue.Queue = queue.Queue(maxsize=self.cfg.spool_max_pending)
        self._worker: threading.Thread | None = None
        self._spool_dir: Path | None = None
        self.draining = False        # 已停止收音、但還在把積壓的語句轉完
        self._starting = False       # 正在載入模型（可能要下載幾百 MB）
        self._start_lock = threading.Lock()
        self._spool_files = 0        # 還沒轉完、還躺在磁碟上的語句數
        self._spool_bytes = 0
        self._spool_lock = threading.Lock()
        self._seq = 0
        self._t0 = 0.0            # 會議開始時刻，各軌的時間戳偏移都以它為基準
        self._lock = threading.Lock()
        self.stats: dict = {"dropped_audio": 0, "dropped_partials": 0, "asr_ms": 0,
                            "utterances": 0, "rejected_nonspeech": 0, "empty_text": 0,
                            "asr_loops": 0}

    # ---------- 軌道 ----------

    def _new_track(self, name: str) -> Track:
        offset = int((time.time() - self._t0) * 1000) if self._t0 else 0
        t = Track(name=name, q=queue.Queue(maxsize=self.cfg.track_queue_blocks),
                  seg=VadSegmenter(self.cfg.vad),
                  clusterer=OnlineClusterer(self.cfg.cluster_threshold,
                                            self.cfg.person_threshold,
                                            key_prefix=TRACK_PREFIX.get(name, "")),
                  offset_ms=offset)
        if self.cfg.enable_diarization:
            t.clusterer.set_people(self.store.person_centroids(EMBED_TAG))
        t.thread = threading.Thread(target=self._vad_loop, args=(t,),
                                    name=f"vad-{name}", daemon=True)
        self._tracks[name] = t
        t.thread.start()
        return t

    def add_track(self, name: str) -> dict:
        """開一條外部餵音訊的軌（目前是瀏覽器分頁音訊）。"""
        if not self.running:
            raise RuntimeError("還沒開始錄音")
        if name in self._tracks:
            return self.tracks_dict()
        self._new_track(name)
        self._emit({"type": "track", "action": "added", "track": name,
                    "tracks": self.tracks_dict()})
        return self.tracks_dict()

    def remove_track(self, name: str) -> dict:
        t = self._tracks.pop(name, None)
        if t is not None:
            t.running = False
            t.q.put(None)
            if t.thread:
                t.thread.join(timeout=4)
            self._emit({"type": "track", "action": "removed", "track": name,
                        "tracks": self.tracks_dict()})
        return self.tracks_dict()

    def push_audio(self, name: str, chunk: np.ndarray) -> None:
        """外部來源餵音訊。呼叫端是 WebSocket，所以這裡不能做任何重活。"""
        t = self._tracks.get(name)
        if t is None or not self.running:
            return
        t.level = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        try:
            t.q.put_nowait(chunk)
        except queue.Full:
            t.dropped += 1
            try:
                t.q.get_nowait()
                t.q.put_nowait(chunk)
            except queue.Empty:
                pass

    # ---------- 音訊暫存（錄音模式） ----------

    @property
    def spool_root(self) -> Path:
        return Path(self.store.path).parent / "spool"

    def cleanup_spool(self) -> int:
        """清掉殘留的暫存音訊。

        正常流程每句轉完就刪，但當機或強制關閉會留下檔案。
        既然專案定案不留音檔，殘留就不該一直躺在磁碟上，所以啟動時清一次。
        """
        root = self.spool_root
        if not root.exists():
            return 0
        n = sum(1 for _ in root.rglob("*.wav"))
        shutil.rmtree(root, ignore_errors=True)
        return n

    def _spool_utterance(self, track: Track, u: Utterance) -> Utterance:
        """把語句音訊寫到磁碟，回傳帶 path、audio 留空的版本。

        寫不進去（磁碟滿、權限問題）就原樣回傳 —— 退回記憶體模式總比掉資料好。
        """
        # 目錄延後建立，這樣錄音中才打開開關也會生效（而不是靜默留在記憶體模式）
        if self._spool_dir is None:
            if self.meeting_id is None:
                return u
            try:
                self._spool_dir = self.spool_root / str(self.meeting_id)
                self._spool_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                self._spool_dir = None
                return u
        try:
            p = self._spool_dir / f"{track.name}-{u.start_ms:09d}-{u.id}.wav"
            sf.write(str(p), u.audio, SR, subtype="FLOAT")
            size = p.stat().st_size
        except Exception:
            self._emit({"type": "error", "where": "spool",
                        "detail": traceback.format_exc()})
            return u
        with self._spool_lock:
            self._spool_files += 1
            self._spool_bytes += size
        self._emit_backlog()
        return replace(u, audio=_EMPTY, path=str(p))

    def _release_spool(self, u: Utterance) -> None:
        """轉寫完就刪掉那一句的音訊。"""
        if not u.path:
            return
        p = Path(u.path)
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        p.unlink(missing_ok=True)
        with self._spool_lock:
            self._spool_files = max(0, self._spool_files - 1)
            self._spool_bytes = max(0, self._spool_bytes - size)
        self._emit_backlog()

    def backlog_dict(self) -> dict:
        with self._spool_lock:
            files, byts = self._spool_files, self._spool_bytes
        return {"pending": self._jobs.qsize(), "files": files, "bytes": byts,
                "draining": self.draining, "spool": self.cfg.spool_audio}

    def _emit_backlog(self) -> None:
        self._emit({"type": "backlog", **self.backlog_dict()})

    def tracks_dict(self) -> dict:
        return {n: {"label": t.label, "level": round(t.level, 4), "dropped": t.dropped,
                    "speakers": sorted(t.clusterer.keys)}
                for n, t in self._tracks.items()}

    # ---------- 生命週期 ----------

    def start(self, title: str = "") -> int:
        if self.running:
            raise RuntimeError("已經在錄音了")
        if self.draining:
            raise RuntimeError("上一場還有語句在轉寫，請等它跑完再開始")
        # running 要等模型全部載完才會是 True，所以不能只看它。
        # 第一次用某個模型可能要下載好幾百 MB，這段時間內重按「開始錄音」
        # 會變成兩條執行緒同時載入同一個模型。
        with self._start_lock:
            if self._starting:
                raise RuntimeError("正在載入模型，請稍候")
            self._starting = True
        try:
            return self._start(title)
        finally:
            self._starting = False

    def _start(self, title: str = "") -> int:
        # 先失敗在這裡，別等到錄了才爆。每個選到的模型都要能載入。
        backend = asr_mod.get_backend(self.cfg.model_id)
        for mid in self.cfg.compare_ids:
            asr_mod.get_backend(mid)
        if self.cfg.enable_diarization:
            self._embedder.load()

        self.meeting_id = self.store.create_meeting(
            asr_model=backend.spec.id, embed_model=EMBED_TAG, embed_dim=EMBED_DIM, title=title)
        self._seq = 0
        self._t0 = time.time()
        self._tracks = {}
        self._spool_files = self._spool_bytes = 0
        self._spool_dir = None      # 第一次要寫暫存檔時才建立
        self.running = True

        self._worker = threading.Thread(target=self._work_loop, name="asr", daemon=True)
        self._worker.start()

        if self.cfg.enable_mic:
            try:
                self._start_mic()
            except Exception:
                self.running = False
                self.store.end_meeting(self.meeting_id)
                raise

        self._emit({"type": "started", "meeting_id": self.meeting_id,
                    "model": self.cfg.model_id, "model_ids": list(self.cfg.model_ids),
                    "embed_model": EMBED_TAG, "tracks": self.tracks_dict()})
        return self.meeting_id

    def _start_mic(self) -> None:
        t = self._new_track(MIC)
        self._mic = MicSource(device=self.cfg.device_index)
        self._mic.start()

        def feed() -> None:
            assert self._mic
            while self.running and t.running:
                chunk = self._mic.read(timeout=0.25)
                if chunk is None or not chunk.size:
                    continue
                t.level = self._mic.level
                t.dropped = self._mic.dropped
                try:
                    t.q.put_nowait(chunk)
                except queue.Full:
                    t.dropped += 1
            t.q.put(None)

        self._mic_thread = threading.Thread(target=feed, name="mic-feed", daemon=True)
        self._mic_thread.start()

    def stop(self) -> None:
        """停止收音。積壓的語句會繼續轉完（draining），不會被丟掉。

        stop() 本身不等待轉寫完成 —— 積壓可能要好幾分鐘，不能讓 HTTP 請求掛在那裡。
        改成背景排空並持續發事件，全部轉完才發 stopped 並結束會議。
        """
        if not self.running:
            return
        self.running = False
        if self._mic:
            self._mic.stop()
        if self._mic_thread:
            self._mic_thread.join(timeout=4)
        for t in list(self._tracks.values()):
            t.running = False
            t.q.put(None)
        for t in list(self._tracks.values()):
            if t.thread:
                t.thread.join(timeout=8)      # flush() 會在這裡補送最後一句
        self._jobs.put(None)

        # 一律在背景排空 —— 佇列長度不含正在跑的那一句，不能拿來判斷有沒有事做
        self.draining = True
        self._emit({"type": "draining", **self.backlog_dict()})
        threading.Thread(target=self._drain, name="drain", daemon=True).start()

    def _drain(self) -> None:
        if self._worker:
            self._worker.join()
        self._finalize()

    def _finalize(self) -> None:
        if self._worker and self._worker.is_alive():
            self._worker.join()
        self.draining = False
        if self.meeting_id is not None:
            self.store.end_meeting(self.meeting_id)
        # 這一場的暫存目錄應該已經空了；殘留代表有語句轉寫失敗，一併清掉
        if self._spool_dir is not None:
            shutil.rmtree(self._spool_dir, ignore_errors=True)
            self._spool_dir = None
        with self._spool_lock:
            self._spool_files = self._spool_bytes = 0
        self._emit({"type": "stopped", "meeting_id": self.meeting_id,
                    "stats": dict(self.stats)})
        self._tracks = {}
        self._mic = None

    def update_config(self, **kw) -> None:
        """錄音中也能改：換模型、調門檻、加減比對模型。下一句就生效。

        ASR 是無狀態呼叫（一句音訊進、一段文字出），每次轉寫都重新查目前的
        設定，所以換模型不需要重啟任何東西。已經轉好的段落不會重跑 ——
        每段都記著自己是哪個模型轉的。
        """
        # 舊版的 model_id / compare_model_id 轉成 model_ids
        if "model_ids" not in kw and ("model_id" in kw or "compare_model_id" in kw):
            ids = [kw.pop("model_id", self.cfg.model_id)]
            extra = kw.pop("compare_model_id", "")
            if extra and extra not in ids:
                ids.append(extra)
            kw["model_ids"] = ids
        if "model_ids" in kw:
            seen, ids = set(), []
            for m in kw["model_ids"] or []:
                if m and m not in seen:
                    seen.add(m)
                    ids.append(m)
            kw["model_ids"] = ids or [asr_mod.DEFAULT_MODEL_ID]

        for k, v in kw.items():
            if k == "vad" and isinstance(v, dict):
                for vk, vv in v.items():
                    if hasattr(self.cfg.vad, vk):
                        setattr(self.cfg.vad, vk, vv)
                for t in self._tracks.values():
                    t.seg.cfg = self.cfg.vad
            elif k in SETTABLE_FIELDS:
                setattr(self.cfg, k, v)
        for t in self._tracks.values():
            t.clusterer.cluster_threshold = self.cfg.cluster_threshold
            t.clusterer.person_threshold = self.cfg.person_threshold
        if self.running and self.meeting_id is not None:
            # 換過模型就更新會議記錄，之後回看歷史才知道這場是哪個模型跑的
            self.store.set_meeting_model(self.meeting_id, self.cfg.model_id)
        self._emit({"type": "config", "config": self.config_dict()})

    def refresh_people(self) -> None:
        """人物庫變動後讓所有軌吃到新聲紋，同一個人下一句就能被認出來。"""
        people = self.store.person_centroids(EMBED_TAG)
        for t in self._tracks.values():
            t.clusterer.set_people(people)

    def config_dict(self) -> dict:
        c = self.cfg
        return {"model_ids": list(c.model_ids), "model_id": c.model_id,
                "compare_ids": c.compare_ids,
                "prompt": c.prompt, "language": c.language, "device_index": c.device_index,
                "enable_mic": c.enable_mic,
                "cluster_threshold": c.cluster_threshold,
                "person_threshold": c.person_threshold,
                "enable_diarization": c.enable_diarization,
                "recluster_every": c.recluster_every,
                "spool_audio": c.spool_audio,
                "speech_gate_db": c.speech_gate_db,
                "vad": {"threshold": c.vad.threshold,
                        "min_silence_ms": c.vad.min_silence_ms,
                        "partial_every_ms": c.vad.partial_every_ms,
                        "max_utterance_ms": c.vad.max_utterance_ms}}

    # ---------- 執行緒 ----------

    def _emit(self, ev: dict) -> None:
        ev.setdefault("t", time.time())
        try:
            self.events.put_nowait(ev)
        except queue.Full:
            pass

    def _vad_loop(self, track: Track) -> None:
        last_level = -1.0
        # 迴圈條件刻意不看 self.running。stop() 第一件事就是把它設成 False，
        # 若以它為條件，佇列裡還沒切句的音訊會被直接拋棄 —— 那就違背了
        # 錄音模式「不丟音訊」的目的。改成把佇列吃到結束標記為止。
        while True:
            try:
                chunk = track.q.get(timeout=0.25)
            except queue.Empty:
                if not (self.running and track.running):
                    break          # 已停止而且佇列也空了
                continue
            if chunk is None:
                break
            try:
                utts = track.seg.push(chunk)
            except Exception:
                self._emit({"type": "error", "where": f"vad:{track.name}",
                            "detail": traceback.format_exc()})
                continue
            for u in utts:
                self._submit(track, u)
            if abs(track.level - last_level) > 0.005:
                last_level = track.level
                self._emit({"type": "level", "track": track.name,
                            "rms": round(track.level, 4), "dropped": track.dropped})
        tail = track.seg.flush()
        if tail is not None:
            self._submit(track, tail)

    def _submit(self, track: Track, u: Utterance) -> None:
        # 暫定稿只在 ASR 空閒時才有意義，落後就直接跳過。永遠不寫磁碟。
        if not u.is_final:
            if not self._jobs.empty():
                self.stats["dropped_partials"] += 1
                return
            try:
                self._jobs.put_nowait((track, u))
            except queue.Full:
                self.stats["dropped_partials"] += 1
            return

        if self.cfg.spool_audio:
            u = self._spool_utterance(track, u)

        if not u.path:
            # 記憶體模式（或寫檔失敗）：等到待處理數降下來才放進去，形成反壓。
            # 代價是上游音訊佇列可能滿而丟資料 —— 這正是錄音模式要避免的事。
            while self.running and self._jobs.qsize() >= MEM_PENDING:
                time.sleep(0.02)
        try:
            self._jobs.put((track, u), timeout=60)
        except queue.Full:
            # 落後到連磁碟都不收了。這是最後一道，只可能在 ASR 長時間遠慢於即時發生。
            self._release_spool(u)
            self._emit({"type": "error", "where": "backlog",
                        "detail": f"待處理已達 {self.cfg.spool_max_pending} 句，"
                                  f"這一句被丟棄。請減少模型數量或換小一點的模型。"})

    def _work_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            track, u = job
            # 佇列裡還有更新的東西時，過期的暫定稿直接跳過
            if not u.is_final and not self._jobs.empty():
                self.stats["dropped_partials"] += 1
                continue
            try:
                self._process(track, u)
            except Exception:
                self._emit({"type": "error", "where": f"asr:{track.name}",
                            "detail": traceback.format_exc()})
            finally:
                # 不管成功或失敗，那一句的音訊都不再需要了
                self._release_spool(u)

    def _process(self, track: Track, u: Utterance) -> None:
        cfg = self.cfg
        # 補上該軌的起始偏移，讓兩軌的時間戳落在同一條時間軸上
        start_ms = u.start_ms + track.offset_ms
        end_ms = u.end_ms + track.offset_ms

        audio = u.audio
        if u.path:
            data, _ = sf.read(u.path, dtype="float32")
            audio = np.asarray(data, dtype=np.float32).reshape(-1)
        if not audio.size:
            return

        if 0 < cfg.speech_gate_db > modulation_depth_db(audio):
            self.stats["rejected_nonspeech"] += 1
            return

        backend = asr_mod.get_backend(cfg.model_id)
        res = backend.transcribe(audio, prompt=cfg.prompt, language=cfg.language)
        self.stats["asr_ms"] = res.latency_ms
        self.stats["asr_loops"] += res.dropped_loops

        if not u.is_final:
            if res.text:
                self._emit({"type": "partial", "track": track.name, "utt": u.id,
                            "text": res.text, "start_ms": start_ms, "end_ms": end_ms,
                            "latency_ms": res.latency_ms, "model": cfg.model_id})
            return

        if not res.text.strip():
            self.stats["empty_text"] += 1
            return

        # 其餘選到的模型逐一跑同一段音訊。序列化執行 —— GPU 本來就一次一個，
        # 而且這裡的目的是比較品質，不是搶速度。
        alts: list[dict] = []
        for mid in cfg.compare_ids:
            try:
                alt = asr_mod.get_backend(mid).transcribe(
                    audio, prompt=cfg.prompt, language=cfg.language)
                alts.append({"model": mid, "text": alt.text, "latency_ms": alt.latency_ms})
            except Exception:
                self._emit({"type": "error", "where": f"compare:{mid}",
                            "detail": traceback.format_exc()})

        vec, assign = None, None
        if cfg.enable_diarization:
            vec = self._embedder.embed(audio)
            if vec is not None:
                assign = track.clusterer.assign(vec, duration_ms=end_ms - start_ms)

        with self._lock:
            self._seq += 1
            seq = self._seq
        key = assign.key if assign else None
        seg_id = self.store.add_segment(
            meeting_id=self.meeting_id, seq=seq, start_ms=start_ms, end_ms=end_ms,
            text=res.text, speaker_key=key, embedding=vec, embed_model=EMBED_TAG,
            track=track.name, lang=res.lang, asr_model=cfg.model_id, alts=alts)

        label = ""
        if assign and assign.person_id:
            # 開集比對命中人物庫：直接把這一群認定成那個人
            self.store.name_speaker(self.meeting_id, key, assign.person_name)
            label = assign.person_name
        elif key:
            for sp in self.store.speakers(self.meeting_id):
                if sp["speaker_key"] == key:
                    label = sp["label"] or sp["person_name"] or ""
                    break

        if key:
            track.seg_ids.append(seg_id)
            track.seg_keys.append(key)
        self.stats["utterances"] += 1

        self._emit({"type": "segment", "id": seg_id, "seq": seq, "track": track.name,
                    "start_ms": start_ms, "end_ms": end_ms, "text": res.text,
                    "speaker_key": key, "speaker_label": label, "lang": res.lang,
                    "model": cfg.model_id, "alts": alts,
                    "latency_ms": res.latency_ms,
                    "audio_ms": u.end_ms - u.start_ms,
                    # 第一位說話者沒有群心可比，score 會是 -1；那時不該顯示分數
                    "spk_score": round(assign.score, 3) if assign and assign.score >= 0 else None,
                    "is_new_speaker": assign.is_new if assign else False,
                    "person_score": round(assign.person_score, 3) if assign else None})

        self._maybe_recluster(track)

    def _maybe_recluster(self, track: Track) -> None:
        """在軌內全域重算分群，追溯修正前面標錯的段落。

        線上分群開頭最不準（群心還沒收斂），這一步是把它救回來的唯一辦法。
        因此 UI 上已輸出的段落必須允許被改寫，不能是 append-only 的捲動列表。
        """
        cfg = self.cfg
        if not cfg.enable_diarization or cfg.recluster_every <= 0:
            return
        track.since_recluster += 1
        if track.since_recluster < cfg.recluster_every or len(track.seg_ids) < 4:
            return
        track.since_recluster = 0
        ids, mat = self.store.meeting_embeddings(self.meeting_id, track=track.name)
        if len(ids) < 4:
            return
        prev = dict(zip(track.seg_ids, track.seg_keys))
        prev_keys = [prev.get(sid, "") for sid in ids]
        durs = self.store.segment_durations(self.meeting_id, ids)
        new_keys = track.clusterer.recluster(mat, prev_keys, durations_ms=durs)
        changed = [(sid, k) for sid, k, old in zip(ids, new_keys, prev_keys) if k != old]
        if not changed:
            return
        self.store.bulk_reassign(changed)
        lookup = dict(zip(ids, new_keys))
        track.seg_keys = [lookup.get(sid, k) for sid, k in zip(track.seg_ids, track.seg_keys)]
        self._emit({"type": "relabel", "track": track.name,
                    "changes": [{"id": sid, "speaker_key": k} for sid, k in changed],
                    "speakers": self.store.speakers(self.meeting_id)})


def audio_devices() -> list[dict]:
    return [{"index": d.index, "name": d.name, "channels": d.channels,
             "default_sr": d.default_sr, "is_default": d.is_default,
             "is_loopback": d.is_loopback}
            for d in list_input_devices()]
