"""本機 HTTP / WebSocket 伺服器。全部只綁 127.0.0.1，不對外。

桌面殼（Tauri/Electron）之後只要指向這個 port 即可，UI 不用重寫。
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from . import asr as asr_mod
from .pipeline import SYSTEM, TRACK_PREFIX, Session, audio_devices
from .spk import EMBED_TAG, OnlineClusterer
from .store import Store

WEB = Path(__file__).parent / "web"
DB_PATH = Path("data/meetings.db")


def create_app(db_path: Path | str = DB_PATH) -> FastAPI:
    app = FastAPI(title="meeting-transcriber")
    store = Store(db_path)
    session = Session(store)
    clients: set[WebSocket] = set()

    async def broadcast(ev: dict) -> None:
        dead = []
        msg = json.dumps(ev, ensure_ascii=False, default=str)
        for ws in list(clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    async def pump() -> None:
        """把工作執行緒丟進 queue 的事件搬到 WebSocket。"""
        loop = asyncio.get_running_loop()
        while True:
            try:
                ev = await loop.run_in_executor(None, session.events.get, True, 0.5)
            except Exception:
                await asyncio.sleep(0.05)
                continue
            if ev:
                await broadcast(ev)

    @app.on_event("startup")
    async def _startup() -> None:
        # 專案定案不留音檔，當機殘留的暫存音訊不該一直躺在磁碟上
        n = session.cleanup_spool()
        if n:
            print(f"  清掉 {n} 個殘留的暫存音訊檔", flush=True)
        app.state.pump = asyncio.create_task(pump())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        session.stop()
        app.state.pump.cancel()

    # ---------- 頁面 ----------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (WEB / "index.html").read_text(encoding="utf-8")

    # ---------- 模型與裝置 ----------

    @app.get("/api/models")
    async def models() -> dict:
        return {"models": asr_mod.catalog(), "default": asr_mod.DEFAULT_MODEL_ID,
                "default_prompt": asr_mod.DEFAULT_PROMPT}

    @app.get("/api/devices")
    async def devices() -> dict:
        return {"devices": audio_devices()}

    @app.post("/api/models/{model_id}/preload")
    async def preload_model(model_id: str) -> dict:
        """先把模型載進記憶體。第一次要下載權重，不能等到按下開始錄音才做。"""
        spec = asr_mod.MODELS_BY_ID.get(model_id)
        if spec is None:
            raise HTTPException(404, f"未知的模型 id: {model_id}")
        await broadcast({"type": "preload", "phase": "start", "model": model_id,
                         "label": spec.label})
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, asr_mod.get_backend, model_id)
        except Exception as e:
            await broadcast({"type": "preload", "phase": "error", "model": model_id,
                             "detail": str(e)})
            raise HTTPException(400, str(e)) from e
        await broadcast({"type": "preload", "phase": "done", "model": model_id,
                         "label": spec.label})
        return {"ok": True}

    install_lock = asyncio.Lock()

    @app.post("/api/models/{model_id}/install")
    async def install_model(model_id: str) -> dict:
        """透過 UI 安裝模型缺少的 Python 套件。

        用 sys.executable -m pip —— 也就是「跑這支伺服器的那個 Python」。
        從 .venv 啟動就裝進 .venv，用系統 Python 啟動就裝進系統，
        不必自己猜該用哪個 pip，也不會裝錯地方。
        """
        spec = asr_mod.MODELS_BY_ID.get(model_id)
        if spec is None:
            raise HTTPException(404, f"未知的模型 id: {model_id}")
        if spec.installed:
            return {"ok": True, "already": True, "models": asr_mod.catalog()}
        if install_lock.locked():
            raise HTTPException(409, "已經有一個安裝在進行中，請等它跑完")

        async with install_lock:
            pkgs = list(spec.pip_packages)
            cmd = [sys.executable, "-m", "pip", "install", *pkgs]
            await broadcast({"type": "install", "phase": "start", "model": model_id,
                             "packages": pkgs, "cmd": " ".join(cmd)})
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            tail: list[str] = []
            assert proc.stdout
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if not line:
                    continue
                tail = (tail + [line])[-40:]
                # pip 的進度條會刷出大量重複的行，只轉發有意義的
                if line.startswith(("Collecting", "Downloading", "Installing",
                                    "Successfully", "Building", "Requirement",
                                    "ERROR", "WARNING")):
                    await broadcast({"type": "install", "phase": "log",
                                     "model": model_id, "line": line})
            rc = await proc.wait()

            # find_spec 會快取，剛裝好的套件要讓它重新找一次
            importlib.invalidate_caches()
            still_missing = spec.missing
            ok = rc == 0 and not still_missing
            await broadcast({"type": "install", "phase": "done", "model": model_id,
                             "ok": ok, "returncode": rc, "missing": still_missing,
                             "models": asr_mod.catalog()})
            if not ok:
                detail = "\n".join(tail[-12:]) or f"pip 結束碼 {rc}"
                if still_missing:
                    detail += f"\n（裝完仍找不到：{', '.join(still_missing)}，可能需要重開伺服器）"
                raise HTTPException(500, detail)
        return {"ok": True, "models": asr_mod.catalog()}

    @app.get("/api/status")
    async def status() -> dict:
        return {"running": session.running, "meeting_id": session.meeting_id,
                "config": session.config_dict(), "stats": session.stats,
                "backlog": session.backlog_dict(), "embed_model": EMBED_TAG}

    @app.post("/api/config")
    async def set_config(req: Request) -> dict:
        body = await req.json()
        try:
            session.update_config(**body)
        except Exception as e:
            raise HTTPException(400, str(e)) from e
        return {"config": session.config_dict()}

    # ---------- 錄音 ----------

    @app.post("/api/start")
    async def start(req: Request) -> dict:
        body = await req.json() if await req.body() else {}
        cfg = {k: v for k, v in body.items() if k != "title"}
        before = session.config_dict()      # 啟動失敗要能還原
        if cfg:
            session.update_config(**cfg)
        try:
            # 模型第一次載入會下載權重，可能要幾十秒；別卡住 event loop
            mid = await asyncio.get_running_loop().run_in_executor(
                None, session.start, body.get("title", ""))
        except Exception as e:
            # 失敗的啟動不能留下副作用。否則選到未安裝的模型後，
            # 設定會卡在一個永遠開不起來的狀態，而 UI 只會忠實顯示它。
            session.update_config(**before)
            raise HTTPException(400, str(e)) from e
        return {"meeting_id": mid}

    @app.post("/api/stop")
    async def stop() -> dict:
        await asyncio.get_running_loop().run_in_executor(None, session.stop)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        clients.add(ws)
        await ws.send_text(json.dumps({"type": "hello", "running": session.running,
                                       "meeting_id": session.meeting_id,
                                       "config": session.config_dict(),
                                       "backlog": session.backlog_dict()},
                                      ensure_ascii=False))
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    @app.websocket("/ws/audio")
    async def ws_audio(ws: WebSocket) -> None:
        """瀏覽器擷取的分頁／系統音訊入口。

        為什麼走瀏覽器：macOS 要收系統音訊，原本得裝虛擬音訊裝置（BlackHole），
        而那需要管理者權限。Chrome 的 getDisplayMedia 可以擷取分頁音訊，
        不必安裝任何東西、不必改系統聲音設定，也不需要管理者權限。
        限制是只抓得到 Chrome 分頁裡的聲音（YouTube、Google Meet 都符合）。

        傳進來的是 16 kHz mono float32 的原始 PCM，前端已經處理好取樣率。
        """
        track = ws.query_params.get("track", SYSTEM)
        await ws.accept()
        if not session.running:
            await ws.send_text(json.dumps({"type": "error", "detail": "還沒開始錄音"}))
            await ws.close()
            return
        try:
            session.add_track(track)
        except Exception as e:
            await ws.send_text(json.dumps({"type": "error", "detail": str(e)},
                                          ensure_ascii=False))
            await ws.close()
            return
        await ws.send_text(json.dumps({"type": "ready", "track": track}))
        try:
            while True:
                buf = await ws.receive_bytes()
                if not buf:
                    continue
                # frombuffer 是零複製的檢視，而 push_audio 之後會被別的執行緒讀，
                # 所以要 copy 一份再交出去。
                session.push_audio(track, np.frombuffer(buf, dtype=np.float32).copy())
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            session.remove_track(track)

    @app.get("/api/tracks")
    async def tracks() -> dict:
        return {"tracks": session.tracks_dict(), "running": session.running,
                "backlog": session.backlog_dict()}

    # ---------- 歷史 ----------

    @app.get("/api/meetings")
    async def meetings(people: str = "") -> dict:
        uids = [x for x in people.split(",") if x]
        return {"meetings": store.list_meetings(uids or None)}

    @app.get("/api/meetings/{mid}")
    async def meeting(mid: int) -> dict:
        m = store.get_meeting(mid)
        if m is None:
            raise HTTPException(404, "找不到這場會議")
        return {"meeting": m, "speakers": store.speakers(mid),
                "segments": [s.__dict__ for s in store.segments(mid)]}

    @app.patch("/api/meetings/{mid}")
    async def patch_meeting(mid: int, req: Request) -> dict:
        body = await req.json()
        if "title" in body:
            store.rename_meeting(mid, body["title"])
        if "pinned" in body:
            store.set_meeting_pinned(mid, bool(body["pinned"]))
        return {"ok": True}

    @app.delete("/api/meetings/{mid}")
    async def delete_meeting(mid: int) -> dict:
        if (session.running or session.draining) and session.meeting_id == mid:
            raise HTTPException(400, "這場還在錄音或收尾中，等它結束再刪")
        store.delete_meeting(mid)
        return {"ok": True}

    @app.get("/api/meetings/{mid}/export")
    async def export_meeting(mid: int, fmt: str = "txt"):
        try:
            if fmt == "json":
                return JSONResponse(store.export_meeting_json(mid))
            return PlainTextResponse(store.export_meeting_text(mid))
        except ValueError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/meetings/{mid}/recluster")
    async def recluster(mid: int, req: Request) -> dict:
        """對已結束的會議重新分群。門檻調不好的舊會議可以事後救。

        每一軌各自重算 —— 麥克風與系統音訊的通道特性不同，跨軌分群
        會把同一個人拆成兩群。
        """
        body = await req.json() if await req.body() else {}
        thr = float(body.get("threshold", session.cfg.cluster_threshold))
        segs = store.segments(mid)
        if not segs:
            raise HTTPException(404, "找不到這場會議")
        tracks = sorted({s.track for s in segs})
        by_id = {s.id: s.speaker_key or "" for s in segs}
        total, loop = 0, asyncio.get_running_loop()
        for track in tracks:
            ids, mat = store.meeting_embeddings(mid, track=track)
            if len(ids) < 2:
                continue
            prev = [by_id.get(i, "") for i in ids]
            durs = store.segment_durations(mid, ids)
            c = OnlineClusterer(thr, session.cfg.person_threshold,
                                key_prefix=TRACK_PREFIX.get(track, ""))
            new = await loop.run_in_executor(None, c.recluster, mat, prev, thr, durs)
            changed = [(i, k) for i, k, o in zip(ids, new, prev) if k != o]
            store.bulk_reassign(changed)
            total += len(changed)
        if not total and all(len(store.meeting_embeddings(mid, track=t)[0]) < 2 for t in tracks):
            raise HTTPException(400, "這場沒有足夠的聲紋向量可以重新分群")
        return {"changed": total, "speakers": store.speakers(mid),
                "segments": [s.__dict__ for s in store.segments(mid)]}

    # ---------- 段落編輯（追溯修正） ----------

    @app.patch("/api/segments/{sid}")
    async def patch_segment(sid: int, req: Request) -> dict:
        body = await req.json()
        if "text" in body:
            store.update_segment_text(sid, body["text"])
        if "speaker_key" in body:
            store.reassign_segment_speaker(sid, body["speaker_key"])
        return {"ok": True}

    # ---------- 人物庫 ----------

    @app.post("/api/meetings/{mid}/speakers/{key}/name")
    async def name_speaker(mid: int, key: str, req: Request) -> dict:
        body = await req.json()
        try:
            r = store.name_speaker(mid, key, body.get("name", ""))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        # 命名完立刻讓進行中的會議吃到新聲紋，同一個人下一句就能被認出來
        session.refresh_people()
        return {**r, "speakers": store.speakers(mid), "people": store.list_people()}

    @app.get("/api/people")
    async def people() -> dict:
        return {"people": store.list_people(), "embed_model": EMBED_TAG}

    @app.patch("/api/people/{pid}")
    async def patch_person(pid: int, req: Request) -> dict:
        body = await req.json()
        store.rename_person(pid, body["name"])
        session.refresh_people()
        return {"people": store.list_people()}

    @app.delete("/api/people/{pid}")
    async def delete_person(pid: int) -> dict:
        store.delete_person(pid)
        session.refresh_people()
        return {"people": store.list_people()}

    @app.get("/api/people/export")
    async def export_people(ids: str = ""):
        person_ids = [int(x) for x in ids.split(",") if x.strip()] or None
        payload = store.export_people(person_ids)
        name = "people-all" if person_ids is None else \
            "people-" + "-".join(str(i) for i in person_ids)
        return JSONResponse(payload, headers={
            "Content-Disposition": f'attachment; filename="{name}.json"'})

    @app.post("/api/people/import/inspect")
    async def inspect_import(req: Request) -> dict:
        """先回報會發生什麼事，名稱撞到的交給使用者決定，不寫任何東西。"""
        body = await req.json()
        try:
            return store.inspect_import(body.get("payload", body))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/people/import")
    async def import_people(req: Request) -> dict:
        body = await req.json()
        try:
            r = store.import_people(body.get("payload", body), body.get("decisions"))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        session.refresh_people()
        return {**r, "people": store.list_people()}

    @app.get("/api/people/{pid}/similar")
    async def similar(pid: int) -> dict:
        """診斷用：這個人和庫裡其他人的相似度。太接近就代表門檻要調高。"""
        cents = store.person_centroids(EMBED_TAG)
        me = next((c for c in cents if c[0] == pid), None)
        if me is None:
            raise HTTPException(404, "找不到這個人，或他的聲紋是別的模型建立的")
        out = [{"id": p, "name": n, "score": round(float(np.dot(me[2], c)), 3)}
               for p, n, c in cents if p != pid]
        return {"others": sorted(out, key=lambda x: -x["score"])}

    app.state.store = store
    app.state.session = session
    return app


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="meeting-transcriber 本機伺服器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8823)
    ap.add_argument("--db", default=str(DB_PATH))
    a = ap.parse_args()
    print(f"  meeting-transcriber → http://{a.host}:{a.port}")
    uvicorn.run(create_app(a.db), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
