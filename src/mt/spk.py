"""說話者辨識：embedding + 線上分群 + 人物庫開集比對。

三件事分開看：
1. embedding —— ECAPA-TDNN 把語音片段轉成 192 維向量，同一人靠近、不同人拉遠。
   與語言無關，所以中英夾雜完全不影響這一層（CLAUDE.md「有利的兩點」之一）。
2. 線上分群 —— 即時只能一句一句往現有群心靠，開頭幾句必然最不準。
3. 開集比對 —— 有了人物庫，熟面孔第一句就能標對；但陌生人誤標成熟人比退回
   「說話者 1」糟得多，所以人物比對的門檻明顯高於分群門檻。

追溯修正的關鍵在 recluster()：線上結果只是暫時的，累積足夠向量後用全域
階層式分群重算一次，把開頭標錯的段落改回來。UI 因此不能是 append-only。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMBED_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
EMBED_DIM = 192
EMBED_VERSION = "ecapa-voxceleb-1"     # 陷阱二：向量一定要帶版本

# 換模型或升版時必須改這個字串，否則舊向量會被當成新向量混用，
# 症狀是比對分數莫名全面下降，而且極難診斷。
EMBED_TAG = f"{EMBED_MODEL}@{EMBED_VERSION}"

# 代號用不設上限的序號。A~Z 只有 26 個，分群失準時很快就用完，
# 而且撞到上限後會退成別的格式，同一場裡出現兩種代號更難看。
def _keys(prefix: str = ""):
    n = 1
    while True:
        yield f"{prefix}{n}"
        n += 1


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_norm(a), _norm(b)))


class Embedder:
    """ECAPA-TDNN 聲紋。第一次呼叫會下載約 80 MB 權重。"""

    def __init__(self, device: str = "cpu", savedir: str | None = None):
        self.device = device
        self.savedir = savedir
        self._m = None
        self._torch = None

    def load(self) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        self._m = EncoderClassifier.from_hparams(
            source=EMBED_MODEL,
            savedir=self.savedir or "models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        if self._m is None:
            self.load()
        if audio.size < 16_000 * 0.4:      # 太短的片段向量不可靠，寧可不給
            return None
        t = self._torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        with self._torch.no_grad():
            e = self._m.encode_batch(t).squeeze().cpu().numpy()
        return _norm(np.asarray(e, dtype=np.float32).reshape(-1))


@dataclass
class Assignment:
    key: str
    score: float
    is_new: bool
    person_id: int | None = None
    person_name: str = ""
    person_score: float = 0.0


class OnlineClusterer:
    """一句一句進來的線上分群。

    cluster_threshold: 與群心的餘弦相似度低於此就開新的說話者。
    person_threshold: 人物庫比對門檻，刻意設得更高（陷阱三）。
    min_new_speaker_ms: 比這短的語句不准開新說話者，只能併進既有的。
    """

    def __init__(self, cluster_threshold: float = 0.45, person_threshold: float = 0.70,
                 person_margin: float = 0.06, key_prefix: str = "",
                 min_new_speaker_ms: int = 2000):
        self.cluster_threshold = cluster_threshold
        self.person_threshold = person_threshold
        self.min_new_speaker_ms = min_new_speaker_ms
        self.person_margin = person_margin      # 與第二名的差距，太接近就不敢認
        # 每一軌各自分群，代號不能撞。麥克風用 A/B/C，系統音訊用 RA/RB/RC。
        # 分軌的理由見 CLAUDE.md：兩個來源的通道特性不同，混在一起分群
        # 會把同一個人拆成兩群。
        self.key_prefix = key_prefix
        self._sums: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._people: list[tuple[int, str, np.ndarray]] = []
        self._pinned: dict[str, int] = {}       # speaker_key -> person_id

    def set_people(self, people: list[tuple[int, str, np.ndarray]]) -> None:
        self._people = people

    @property
    def keys(self) -> list[str]:
        return sorted(self._sums)

    def centroid(self, key: str) -> np.ndarray | None:
        if key not in self._sums or self._counts[key] == 0:
            return None
        return _norm(self._sums[key] / self._counts[key])

    def _next_key(self) -> str:
        for k in _keys(self.key_prefix):
            if k not in self._sums:
                return k
        raise AssertionError("unreachable")

    def assign(self, vec: np.ndarray, duration_ms: int | None = None) -> Assignment:
        vec = _norm(vec)
        best_key, best = "", -1.0
        for k in self._sums:
            s = cosine(vec, self.centroid(k))
            if s > best:
                best_key, best = k, s

        too_short = duration_ms is not None and duration_ms < self.min_new_speaker_ms
        if best_key and (best >= self.cluster_threshold or too_short):
            key, is_new = best_key, False
        else:
            key, is_new, best = self._next_key(), True, best
            self._sums[key] = np.zeros_like(vec)
            self._counts[key] = 0

        self._sums[key] += vec
        self._counts[key] += 1

        pid, pname, pscore = self._match_person(key, vec)
        return Assignment(key=key, score=float(best), is_new=is_new,
                          person_id=pid, person_name=pname, person_score=pscore)

    def _match_person(self, key: str, vec: np.ndarray) -> tuple[int | None, str, float]:
        if not self._people:
            return None, "", 0.0
        # 這個群已經認定過某人就沿用，不要每句話跳來跳去
        if key in self._pinned:
            pid = self._pinned[key]
            for p_id, name, c in self._people:
                if p_id == pid:
                    return p_id, name, cosine(vec, c)
        c = self.centroid(key)
        target = c if c is not None else vec
        scored = sorted(((cosine(target, cen), pid, name)
                         for pid, name, cen in self._people), reverse=True)
        top = scored[0]
        runner = scored[1][0] if len(scored) > 1 else -1.0
        # 兩道保險：分數要夠高，且要明顯贏過第二名。
        # 寧可留在未命名狀態，也不要把陌生人叫成 Daniel。
        if top[0] >= self.person_threshold and (top[0] - runner) >= self.person_margin:
            self._pinned[key] = top[1]
            return top[1], top[2], top[0]
        return None, "", top[0]

    # ---------- 追溯修正 ----------

    def recluster(self, vectors: np.ndarray, prev_keys: list[str],
                  threshold: float | None = None,
                  durations_ms: list[int] | None = None) -> list[str]:
        """對整場會議的向量重新分群，回傳每段新的 speaker_key。

        分群本質上是全域操作，線上版本只是妥協。這裡用階層式分群重算一次，
        再把新群對回舊代號（取重疊最多的），讓已命名的說話者代號不會跳掉。

        只有夠長的段落參與決定群數，短的最後併進最像的那一群。
        """
        n = len(vectors)
        if n == 0:
            return []
        thr = self.cluster_threshold if threshold is None else threshold
        V = np.stack([_norm(v) for v in vectors])
        long_idx = [i for i in range(n)] if durations_ms is None else \
            [i for i in range(n) if durations_ms[i] >= self.min_new_speaker_ms]
        if not long_idx:                    # 全部都太短，那就全部一起看
            long_idx = list(range(n))
        short_idx = [i for i in range(n) if i not in set(long_idx)]

        # 用群心比，跟 assign() 同一套標準
        def cen(idxs: list[int]) -> np.ndarray:
            return _norm(V[idxs].mean(axis=0))

        clusters: list[list[int]] = [[i] for i in long_idx]
        while len(clusters) > 1:
            best, pair = -np.inf, None
            for a in range(len(clusters)):
                ca = cen(clusters[a])
                for b in range(a + 1, len(clusters)):
                    s = float(ca @ cen(clusters[b]))
                    if s > best:
                        best, pair = s, (a, b)
            if pair is None or best < thr:
                break
            a, b = pair
            clusters[a] = clusters[a] + clusters[b]
            clusters.pop(b)

        # 太短的段落不參與決定群數，最後併進最像的那一群
        for i in short_idx:
            ci = max(range(len(clusters)), key=lambda c: float(V[i] @ cen(clusters[c])))
            clusters[ci].append(i)

        # 新群對回舊代號：以重疊段數最多者優先，避免代號洗牌
        order = sorted(range(len(clusters)), key=lambda i: -len(clusters[i]))
        assigned: dict[int, str] = {}
        used: set[str] = set()
        for ci in order:
            votes: dict[str, int] = {}
            for i in clusters[ci]:
                k = prev_keys[i] if i < len(prev_keys) else None
                if k:
                    votes[k] = votes.get(k, 0) + 1
            for k, _ in sorted(votes.items(), key=lambda kv: -kv[1]):
                if k not in used:
                    assigned[ci] = k
                    used.add(k)
                    break
        fresh = (k for k in _keys(self.key_prefix) if k not in used)
        for ci in order:
            if ci not in assigned:
                assigned[ci] = next(fresh, f"{self.key_prefix}S{ci + 1}")

        out = [""] * n
        for ci, idxs in enumerate(clusters):
            for i in idxs:
                out[i] = assigned[ci]

        # 重建群心，讓後續的線上分群接在重算後的結果上
        self._sums, self._counts = {}, {}
        for i, k in enumerate(out):
            self._sums.setdefault(k, np.zeros(V.shape[1], dtype=np.float32))
            self._sums[k] += V[i]
            self._counts[k] = self._counts.get(k, 0) + 1
        return out
