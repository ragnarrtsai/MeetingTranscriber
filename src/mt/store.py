"""SQLite 儲存層。

兩個設計約束來自 CLAUDE.md：
- 陷阱一：每個說話者群的 embedding 必須在會議進行時就落地，不能等改名時才回頭找。
  因此 segment 一產生就連同向量寫入，不留音檔也不影響後續建立聲紋。
- 陷阱二：向量一定要記下嵌入模型的名稱與版本，換模型後舊向量無法轉換，
  沒有版本欄位會變成靜默失敗（比對分數莫名下降）。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT    NOT NULL DEFAULT '',
    started_at    REAL    NOT NULL,
    ended_at      REAL,
    asr_model     TEXT    NOT NULL,
    embed_model   TEXT    NOT NULL,
    embed_dim     INTEGER NOT NULL,
    notes         TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    start_ms    INTEGER NOT NULL,
    end_ms      INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    track       TEXT    NOT NULL DEFAULT 'mic',
    speaker_key TEXT,
    person_id   INTEGER REFERENCES people(id) ON DELETE SET NULL,
    lang        TEXT NOT NULL DEFAULT '',
    asr_model   TEXT NOT NULL DEFAULT '',
    -- 舊版只支援一個比對模型時的欄位。新資料寫進 segment_alts，
    -- 這兩欄留著是為了讓舊的會議記錄還讀得出來。
    alt_text    TEXT NOT NULL DEFAULT '',
    alt_model   TEXT NOT NULL DEFAULT '',
    edited      INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_meeting ON segments(meeting_id, seq);

-- 並排比對：同一句話由其他模型轉寫的結果，一個模型一列。
-- 中英夾雜是本專案最大風險，而它沒辦法靠查文件解決 —— 讓多個模型的輸出
-- 並列在同一句話下面，是判斷品質最省力的方式。
CREATE TABLE IF NOT EXISTS segment_alts (
    segment_id  INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    text        TEXT NOT NULL,
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (segment_id, model)
);

-- 逐段向量。刻意存「每段」而非只存群心，因為改名後要重算群心、
-- 也因為背景重新分群需要原始向量才能追溯修正早期的錯誤標記。
CREATE TABLE IF NOT EXISTS segment_embeddings (
    segment_id  INTEGER PRIMARY KEY REFERENCES segments(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS speakers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker_key TEXT NOT NULL,          -- 'A' / 'B' / 'C'，分群的天然輸出
    label       TEXT NOT NULL DEFAULT '',
    person_id   INTEGER REFERENCES people(id) ON DELETE SET NULL,
    track       TEXT NOT NULL DEFAULT 'mic',
    UNIQUE (meeting_id, speaker_key)
);

-- 人物庫：跨會議的長久記憶。
CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    embed_model TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS person_vectors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    src_meeting INTEGER,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pv_person ON person_vectors(person_id);
"""


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _unblob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


@dataclass
class Segment:
    id: int
    seq: int
    start_ms: int
    end_ms: int
    text: str
    track: str
    speaker_key: str | None
    speaker_label: str
    lang: str
    asr_model: str = ""
    # 其他模型對同一句話的輸出：[{"model": ..., "text": ..., "latency_ms": ...}]
    alts: list[dict] = field(default_factory=list)
    edited: bool = False
    has_embedding: bool = False


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # ---------- meetings ----------

    def create_meeting(self, asr_model: str, embed_model: str, embed_dim: int,
                       title: str = "") -> int:
        now = time.time()
        cur = self._db.execute(
            "INSERT INTO meetings (title, started_at, asr_model, embed_model, embed_dim)"
            " VALUES (?,?,?,?,?)",
            (title or time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
             now, asr_model, embed_model, embed_dim),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def end_meeting(self, meeting_id: int) -> None:
        self._db.execute("UPDATE meetings SET ended_at=? WHERE id=?", (time.time(), meeting_id))
        self._db.commit()

    def set_meeting_model(self, meeting_id: int, asr_model: str) -> None:
        """錄音中換模型時更新，之後回看歷史才知道這場是哪個模型跑的。"""
        self._db.execute("UPDATE meetings SET asr_model=? WHERE id=?", (asr_model, meeting_id))
        self._db.commit()

    def rename_meeting(self, meeting_id: int, title: str) -> None:
        self._db.execute("UPDATE meetings SET title=? WHERE id=?", (title, meeting_id))
        self._db.commit()

    def delete_meeting(self, meeting_id: int) -> None:
        self._db.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
        self._db.commit()

    def list_meetings(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT m.*,"
            " (SELECT COUNT(*) FROM segments s WHERE s.meeting_id=m.id) AS n_segments,"
            " (SELECT COUNT(*) FROM speakers k WHERE k.meeting_id=m.id) AS n_speakers"
            " FROM meetings m ORDER BY m.started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_meeting(self, meeting_id: int) -> dict | None:
        r = self._db.execute("SELECT * FROM meetings WHERE id=?", (meeting_id,)).fetchone()
        return dict(r) if r else None

    # ---------- segments ----------

    def add_segment(self, meeting_id: int, seq: int, start_ms: int, end_ms: int,
                    text: str, speaker_key: str | None, embedding: np.ndarray | None,
                    embed_model: str, track: str = "mic", lang: str = "",
                    asr_model: str = "", alts: Sequence[dict] | None = None) -> int:
        cur = self._db.execute(
            "INSERT INTO segments (meeting_id, seq, start_ms, end_ms, text, track,"
            " speaker_key, lang, asr_model, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (meeting_id, seq, start_ms, end_ms, text, track, speaker_key, lang,
             asr_model, time.time()),
        )
        seg_id = int(cur.lastrowid)
        for a in alts or []:
            self._db.execute(
                "INSERT OR REPLACE INTO segment_alts (segment_id, model, text, latency_ms)"
                " VALUES (?,?,?,?)",
                (seg_id, a["model"], a.get("text", ""), int(a.get("latency_ms", 0))))
        # 向量與逐字稿在同一個 transaction 落地。若這裡偷懶只寫文字，
        # 使用者改名時就沒有聲紋可用 —— 音檔那時已經不存在了。
        if embedding is not None:
            self._db.execute(
                "INSERT INTO segment_embeddings (segment_id, model, dim, vector) VALUES (?,?,?,?)",
                (seg_id, embed_model, int(embedding.shape[-1]), _blob(embedding)),
            )
        if speaker_key:
            self._db.execute(
                "INSERT OR IGNORE INTO speakers (meeting_id, speaker_key, track) VALUES (?,?,?)",
                (meeting_id, speaker_key, track),
            )
        self._db.commit()
        return seg_id

    def update_segment_text(self, segment_id: int, text: str) -> None:
        self._db.execute("UPDATE segments SET text=?, edited=1 WHERE id=?", (text, segment_id))
        self._db.commit()

    def reassign_segment_speaker(self, segment_id: int, speaker_key: str) -> None:
        row = self._db.execute("SELECT meeting_id, track FROM segments WHERE id=?",
                               (segment_id,)).fetchone()
        if row is None:
            return
        self._db.execute("INSERT OR IGNORE INTO speakers (meeting_id, speaker_key, track)"
                         " VALUES (?,?,?)", (row["meeting_id"], speaker_key, row["track"]))
        self._db.execute("UPDATE segments SET speaker_key=? WHERE id=?", (speaker_key, segment_id))
        self._db.commit()

    def bulk_reassign(self, pairs: Iterable[tuple[int, str]]) -> None:
        """背景重新分群用：一次改掉多段的說話者標記（追溯修正）。"""
        pairs = list(pairs)
        if not pairs:
            return
        for seg_id, key in pairs:
            self._db.execute("UPDATE segments SET speaker_key=? WHERE id=?", (key, seg_id))
        self._db.commit()

    def segments(self, meeting_id: int) -> list[Segment]:
        rows = self._db.execute(
            "SELECT s.*, COALESCE(NULLIF(k.label,''), p.name, '') AS speaker_label,"
            " (e.segment_id IS NOT NULL) AS has_embedding"
            " FROM segments s"
            " LEFT JOIN speakers k ON k.meeting_id=s.meeting_id AND k.speaker_key=s.speaker_key"
            " LEFT JOIN people p ON p.id=k.person_id"
            " LEFT JOIN segment_embeddings e ON e.segment_id=s.id"
            " WHERE s.meeting_id=? ORDER BY s.seq",
            (meeting_id,),
        ).fetchall()
        alts: dict[int, list[dict]] = {}
        for a in self._db.execute(
            "SELECT a.* FROM segment_alts a JOIN segments s ON s.id=a.segment_id"
            " WHERE s.meeting_id=? ORDER BY a.model", (meeting_id,)
        ).fetchall():
            alts.setdefault(int(a["segment_id"]), []).append(
                {"model": a["model"], "text": a["text"], "latency_ms": a["latency_ms"]})

        out = []
        for r in rows:
            got = alts.get(r["id"], [])
            # 舊資料的比對結果存在 alt_text/alt_model，沒有 segment_alts 就退回去讀
            if not got and r["alt_model"]:
                got = [{"model": r["alt_model"], "text": r["alt_text"], "latency_ms": 0}]
            out.append(Segment(
                id=r["id"], seq=r["seq"], start_ms=r["start_ms"], end_ms=r["end_ms"],
                text=r["text"], track=r["track"], speaker_key=r["speaker_key"],
                speaker_label=r["speaker_label"] or "", lang=r["lang"],
                asr_model=r["asr_model"], alts=got,
                edited=bool(r["edited"]), has_embedding=bool(r["has_embedding"])))
        return out

    def meeting_embeddings(self, meeting_id: int,
                           track: str | None = None) -> tuple[list[int], np.ndarray]:
        """取整場（或單一軌）的向量。

        重新分群必須限定在同一軌內：麥克風與系統音訊的通道特性不同，
        跨軌分群會把同一個人拆成兩群。
        """
        sql = ("SELECT e.segment_id, e.vector FROM segment_embeddings e"
               " JOIN segments s ON s.id=e.segment_id WHERE s.meeting_id=?")
        args: tuple = (meeting_id,)
        if track is not None:
            sql += " AND s.track=?"
            args += (track,)
        rows = self._db.execute(sql + " ORDER BY s.seq", args).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [int(r["segment_id"]) for r in rows]
        mat = np.stack([_unblob(r["vector"]) for r in rows])
        return ids, mat

    # ---------- speakers ----------

    def speakers(self, meeting_id: int) -> list[dict]:
        rows = self._db.execute(
            "SELECT k.*, p.name AS person_name,"
            " (SELECT COUNT(*) FROM segments s WHERE s.meeting_id=k.meeting_id"
            "   AND s.speaker_key=k.speaker_key) AS n_segments"
            " FROM speakers k LEFT JOIN people p ON p.id=k.person_id"
            " WHERE k.meeting_id=? ORDER BY k.speaker_key",
            (meeting_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def speaker_vectors(self, meeting_id: int, speaker_key: str) -> np.ndarray:
        rows = self._db.execute(
            "SELECT e.vector FROM segment_embeddings e JOIN segments s ON s.id=e.segment_id"
            " WHERE s.meeting_id=? AND s.speaker_key=?",
            (meeting_id, speaker_key),
        ).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([_unblob(r["vector"]) for r in rows])

    # ---------- people（人物庫） ----------

    def name_speaker(self, meeting_id: int, speaker_key: str, name: str) -> dict:
        """把「A」改成「Daniel」：該群的所有向量存進人物庫。

        這是 CLAUDE.md 定案的聲紋建立方式 —— 事後編輯已記錄的人物。
        """
        name = name.strip()
        meeting = self.get_meeting(meeting_id)
        if meeting is None:
            raise ValueError(f"meeting {meeting_id} not found")
        if not name:
            self._db.execute("UPDATE speakers SET label='', person_id=NULL"
                             " WHERE meeting_id=? AND speaker_key=?", (meeting_id, speaker_key))
            self._db.commit()
            return {"person_id": None, "name": ""}

        vecs = self.speaker_vectors(meeting_id, speaker_key)
        model, dim = meeting["embed_model"], meeting["embed_dim"]
        now = time.time()

        row = self._db.execute("SELECT * FROM people WHERE name=?", (name,)).fetchone()
        if row is None:
            cur = self._db.execute(
                "INSERT INTO people (name, embed_model, dim, created_at, updated_at)"
                " VALUES (?,?,?,?,?)", (name, model, dim, now, now))
            person_id = int(cur.lastrowid)
        else:
            person_id = int(row["id"])
            if row["embed_model"] != model:
                # 換過模型：舊向量不可用也無法轉換，這是已知且接受的代價，
                # 但必須明講而不是默默混用兩種模型的向量。
                raise ValueError(
                    f"人物「{name}」的聲紋是用 {row['embed_model']} 建立的，"
                    f"目前是 {model}。請先重建該人物的聲紋。")
            self._db.execute("UPDATE people SET updated_at=? WHERE id=?", (now, person_id))

        for v in vecs:
            self._db.execute(
                "INSERT INTO person_vectors (person_id, model, dim, vector, src_meeting, created_at)"
                " VALUES (?,?,?,?,?,?)", (person_id, model, dim, _blob(v), meeting_id, now))

        self._db.execute("INSERT OR IGNORE INTO speakers (meeting_id, speaker_key) VALUES (?,?)",
                         (meeting_id, speaker_key))
        self._db.execute("UPDATE speakers SET label=?, person_id=? WHERE meeting_id=? AND speaker_key=?",
                         (name, person_id, meeting_id, speaker_key))
        self._db.commit()
        return {"person_id": person_id, "name": name, "n_vectors_added": int(len(vecs))}

    def list_people(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM person_vectors v WHERE v.person_id=p.id) AS n_vectors,"
            " (SELECT COUNT(DISTINCT v.src_meeting) FROM person_vectors v WHERE v.person_id=p.id)"
            "   AS n_meetings"
            " FROM people p ORDER BY p.name"
        ).fetchall()
        return [dict(r) for r in rows]

    def rename_person(self, person_id: int, name: str) -> None:
        self._db.execute("UPDATE people SET name=?, updated_at=? WHERE id=?",
                         (name.strip(), time.time(), person_id))
        self._db.execute("UPDATE speakers SET label=? WHERE person_id=?", (name.strip(), person_id))
        self._db.commit()

    def delete_person(self, person_id: int) -> None:
        self._db.execute("DELETE FROM people WHERE id=?", (person_id,))
        self._db.commit()

    def person_centroids(self, embed_model: str) -> list[tuple[int, str, np.ndarray]]:
        """人物庫的比對用群心。只回傳同一個嵌入模型建立的人物。"""
        out: list[tuple[int, str, np.ndarray]] = []
        for row in self._db.execute(
            "SELECT id, name FROM people WHERE embed_model=?", (embed_model,)
        ).fetchall():
            vr = self._db.execute(
                "SELECT vector FROM person_vectors WHERE person_id=?", (row["id"],)).fetchall()
            if not vr:
                continue
            mat = np.stack([_unblob(x["vector"]) for x in vr])
            c = mat.mean(axis=0)
            n = np.linalg.norm(c)
            out.append((int(row["id"]), row["name"], c / n if n else c))
        return out

    # ---------- 匯出／匯入 ----------

    EXPORT_VERSION = 1

    def export_people(self, person_ids: Sequence[int] | None = None) -> dict:
        """自訂格式。單人匯出是必要能力，全庫只是 person_ids=None 的特例。"""
        if person_ids is None:
            rows = self._db.execute("SELECT * FROM people ORDER BY name").fetchall()
        else:
            qs = ",".join("?" * len(person_ids))
            rows = self._db.execute(
                f"SELECT * FROM people WHERE id IN ({qs}) ORDER BY name", tuple(person_ids)
            ).fetchall()

        people = []
        for r in rows:
            vr = self._db.execute(
                "SELECT vector, src_meeting, created_at FROM person_vectors WHERE person_id=?",
                (r["id"],)).fetchall()
            people.append({
                "name": r["name"],
                "embed_model": r["embed_model"],   # 陷阱二：版本欄位不可省
                "dim": r["dim"],
                "created_at": r["created_at"],
                "vectors": [_unblob(x["vector"]).round(6).tolist() for x in vr],
            })
        return {
            "format": "meeting-transcriber/people",
            "version": self.EXPORT_VERSION,
            "exported_at": time.time(),
            "people": people,
        }

    def import_people(self, payload: dict, merge: bool = True) -> dict:
        if payload.get("format") != "meeting-transcriber/people":
            raise ValueError("不是本工具的人物庫匯出檔")
        if int(payload.get("version", 0)) > self.EXPORT_VERSION:
            raise ValueError(f"匯出檔版本 {payload['version']} 比目前支援的 "
                             f"{self.EXPORT_VERSION} 新，請先更新程式")
        added, skipped, now = [], [], time.time()
        for p in payload.get("people", []):
            name, model, dim = p["name"], p["embed_model"], int(p["dim"])
            row = self._db.execute("SELECT * FROM people WHERE name=?", (name,)).fetchone()
            if row is not None:
                if not merge or row["embed_model"] != model:
                    skipped.append({"name": name, "reason":
                                    "已存在且嵌入模型不同" if row["embed_model"] != model else "已存在"})
                    continue
                person_id = int(row["id"])
            else:
                cur = self._db.execute(
                    "INSERT INTO people (name, embed_model, dim, created_at, updated_at)"
                    " VALUES (?,?,?,?,?)", (name, model, dim, p.get("created_at", now), now))
                person_id = int(cur.lastrowid)
            for v in p.get("vectors", []):
                self._db.execute(
                    "INSERT INTO person_vectors (person_id, model, dim, vector, src_meeting, created_at)"
                    " VALUES (?,?,?,?,NULL,?)",
                    (person_id, model, dim, _blob(np.asarray(v, dtype=np.float32)), now))
            added.append(name)
        self._db.commit()
        return {"added": added, "skipped": skipped}

    def export_meeting_text(self, meeting_id: int) -> str:
        m = self.get_meeting(meeting_id)
        if m is None:
            raise ValueError("meeting not found")
        lines = [f"# {m['title']}", ""]
        for s in self.segments(meeting_id):
            who = s.speaker_label or s.speaker_key or "?"
            ts = f"{s.start_ms // 60000:02d}:{s.start_ms // 1000 % 60:02d}"
            lines.append(f"[{ts}] {who}: {s.text}")
        return "\n".join(lines) + "\n"

    def export_meeting_json(self, meeting_id: int) -> dict:
        m = self.get_meeting(meeting_id)
        if m is None:
            raise ValueError("meeting not found")
        return {
            "format": "meeting-transcriber/meeting",
            "version": self.EXPORT_VERSION,
            "meeting": m,
            "speakers": self.speakers(meeting_id),
            "segments": [s.__dict__ for s in self.segments(meeting_id)],
        }
