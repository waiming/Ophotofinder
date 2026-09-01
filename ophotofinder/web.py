"""Flask web UI.

Safety rules that are enforced here, not merely documented:

* ``/api/photo`` serves a file only when its absolute path is already in the
  index. An arbitrary path -- however well-formed -- is refused.
* The folder picker can only walk the user's home directory and mounted
  volumes under /Volumes.
* Exactly one indexing run at a time, enforced by the same on-disk lock the
  CLI uses, so a web run and a terminal run cannot both write to Chroma.
"""
from __future__ import annotations

import atexit
import signal
import threading
import time

import requests
from pathlib import Path

from urllib.parse import quote

from flask import Flask, abort, jsonify, render_template, request, send_file

from . import ollama, settings
from .config import THUMB_DIR
from .imaging import path_id, thumbnail_path
from .indexer import index_folders
from .watcher import Watcher
from .lock import IndexBusy, read_holder
from .scan import list_subdirs, picker_roots, scan, within_picker_roots
from .search import Searcher, answer, build_where
from .store import Store


class IndexJob:
    """State of the single permitted indexing run."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        # Set at shutdown so an in-flight run stops at the next photo rather than
        # being torn down mid-computation.
        self.cancel = threading.Event()
        self.reset()

    def reset(self):
        self.running = False
        self.mode = "index"
        self.cancel.clear()
        self.folders: list[str] = []
        self.done = 0
        self.total = 0
        self.current = ""
        self.log: list[dict] = []
        self.notes: list[str] = []
        self.summary: dict | None = None
        self.error = ""
        self.started = 0.0

    def is_running(self) -> bool:
        return self.running or read_holder() is not None

    def snapshot(self) -> dict:
        holder = read_holder()
        return {
            "running": self.running,
            "mode": getattr(self, "mode", "index"),
            "locked_by_other": bool(holder and not self.running),
            "holder": holder,
            "folders": self.folders,
            "done": self.done,
            "total": self.total,
            "current": self.current,
            "log": self.log[-60:],
            "notes": self.notes,
            "summary": self.summary,
            "error": self.error,
            "elapsed": round(time.time() - self.started, 1) if self.started else 0,
        }


def create_app(device: str | None = None) -> Flask:
    app = Flask(__name__)
    store = Store()
    searcher = Searcher(store, device=device)
    job = IndexJob()

    def _watch_index(folders, **kw):
        return index_folders(folders, device=device, store=store, **kw)

    def _watch_upgrade(**kw):
        from .indexer import upgrade_index
        return upgrade_index(device=device, store=store, **kw)

    watcher = Watcher(_watch_index, _watch_upgrade)
    watcher.start()

    def _shutdown():
        """Let an in-flight run finish its current photo instead of being killed."""
        watcher.stop()
        job.cancel.set()
        t = job.thread
        if t and t.is_alive():
            t.join(timeout=20)

    atexit.register(_shutdown)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            _prev = signal.getsignal(_sig)

            def _handler(signum, frame, _prev=_prev):
                _shutdown()
                if callable(_prev):
                    _prev(signum, frame)
                else:
                    raise SystemExit(0)

            signal.signal(_sig, _handler)
        except ValueError:
            pass                  # not on the main thread

    served: set[str] = set(store.indexed_paths())
    served_lock = threading.Lock()

    def refresh_served():
        with served_lock:
            served.clear()
            served.update(store.indexed_paths())

    def is_served(path: str) -> bool:
        with served_lock:
            if path in served:
                return True
        refresh_served()          # a run may have added it since we last looked
        with served_lock:
            return path in served

    # ------------------------------------------------------------------ views

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/status")
    def api_status():
        counts = store.counts()
        try:
            models = [
                {"name": m.name, "vision": m.has_vision, "broken": m.is_broken,
                 "ctx": m.context_length, "params": m.parameter_count,
                 "caps": m.capabilities, "size": m.size, "fit": m.fits()}
                for m in ollama.inventory()
            ]
            ollama_up = True
        except Exception:
            models, ollama_up = [], False
        auto_vision = ""
        try:
            auto_vision = ollama.pick_vision_model().name
        except Exception:
            pass
        auto_text = ollama.pick_text_model() if ollama_up else None
        return jsonify({
            "counts": counts,
            "caption_health": store.caption_health(),
            "folders": store.folders(),
            "ollama_up": ollama_up,
            "ollama_host": ollama.host(),
            "models": models,
            "auto_vision": auto_vision,
            "auto_text": auto_text.name if auto_text else "",
            "settings_text_model": settings.get("text_model", ""),
            "settings_vision_model": settings.get("vision_model", ""),
            "indexing": job.snapshot(),
            "watcher": watcher.status(),
        })

    @app.post("/api/search")
    def api_search():
        body = request.get_json(force=True, silent=True) or {}
        q = (body.get("query") or "").strip()
        if not q:
            return jsonify({"error": "empty query"}), 400
        where = build_where(
            body.get("year") or None, body.get("camera") or None,
            body.get("folder") or None,
            True if body.get("with_text") else None,
        )
        mode = body.get("mode", "both")
        hits = searcher.search(
            q, k=int(body.get("k", 24)), where=where,
            use_clip=mode in ("both", "clip"), use_text=mode in ("both", "text"),
        )
        out = {"hits": [{**h.as_dict(), "thumb": f"/api/thumb/{h.id}"} for h in hits]}
        if body.get("answer"):
            try:
                out["answer"] = answer(
                    q, hits,
                    model=body.get("text_model") or settings.get("text_model") or None)
            except Exception as e:
                out["answer_error"] = str(e)
        return jsonify(out)

    @app.get("/api/thumb/<pid>")
    def api_thumb(pid: str):
        if not pid.isalnum():
            abort(400)
        p = THUMB_DIR / f"{pid}.jpg"
        if not p.is_file():
            abort(404)
        return send_file(p, mimetype="image/jpeg")

    @app.get("/api/photo")
    def api_photo():
        """Serve an original -- but only if that exact path is in the index."""
        raw = request.args.get("path", "")
        if not raw:
            abort(400)
        try:
            path = Path(raw).resolve(strict=True)
        except (OSError, RuntimeError):
            abort(404)
        if not is_served(str(path)):
            abort(403, "that file is not in the ophotofinder index")
        if path.suffix.lower() in {".heic", ".heif", ".avif"}:
            # Browsers cannot render HEIC; hand back the cached upright thumbnail.
            t = thumbnail_path(path)
            if t.is_file():
                return send_file(t, mimetype="image/jpeg")
        return send_file(path)

    @app.get("/api/library")
    def api_library():
        """Browse everything in the index, with all the signals recorded per photo."""
        try:
            offset = max(0, int(request.args.get("offset", 0)))
            limit = min(500, max(1, int(request.args.get("limit", 60))))
        except ValueError:
            return jsonify({"error": "bad offset/limit"}), 400

        where = build_where(
            request.args.get("year") or None,
            request.args.get("camera") or None,
            request.args.get("folder") or None,
            True if request.args.get("with_text") == "1" else None,
        )
        recs = store.all_records_full(where)

        status = request.args.get("caption_status") or ""
        if status:
            recs = [r for r in recs if r.get("caption_status") == status]

        needle = (request.args.get("q") or "").strip().lower()
        if needle:
            # Every text field a photo carries, including the second reader's
            # output -- text in scripts Tesseract cannot read lives only there.
            fields = ("filename", "caption", "ocr", "ocr_alt", "camera",
                      "folder", "taken_at", "document")

            def hit(r):
                return any(needle in str(r.get(k, "")).lower() for k in fields)
            recs = [r for r in recs if hit(r)]

        sort = request.args.get("sort", "taken_desc")
        keyfns = {
            "taken_desc": (lambda r: r.get("taken_at", ""), True),
            "taken_asc": (lambda r: r.get("taken_at", ""), False),
            "name": (lambda r: str(r.get("filename", "")).lower(), False),
            "indexed_desc": (lambda r: r.get("indexed_at", ""), True),
            "folder": (lambda r: (str(r.get("folder", "")), str(r.get("filename", ""))), False),
        }
        keyfn, rev = keyfns.get(sort, keyfns["taken_desc"])
        recs.sort(key=keyfn, reverse=rev)

        total = len(recs)
        page = recs[offset:offset + limit]
        for r in page:
            r["thumb"] = f"/api/thumb/{r['id']}"
            r["photo"] = "/api/photo?path=" + quote(str(r.get("path", "")), safe="")
            r.pop("signature", None)
        return jsonify({
            "total": total, "offset": offset, "limit": limit, "records": page,
            "facets": {
                "folders": store.folders(),
                "cameras": store.cameras(),
                "caption_health": store.caption_health(),
            },
        })

    # ------------------------------------------------------------ folder picker

    @app.get("/api/browse")
    def api_browse():
        raw = request.args.get("path", "")
        if not raw:
            return jsonify({
                "path": "", "parent": None,
                "roots": [{"name": p.name or str(p), "path": str(p)} for p in picker_roots()],
                "dirs": [],
            })
        path = Path(raw).expanduser()
        if not within_picker_roots(path):
            return jsonify({"error": "outside your home directory and mounted volumes"}), 403
        path = path.resolve()
        if not path.is_dir():
            return jsonify({"error": "not a directory"}), 404
        parent = str(path.parent) if within_picker_roots(path.parent) else None
        rep = scan([path], recursive=False)
        deep = scan([path], recursive=True)
        return jsonify({
            "path": str(path),
            "parent": parent,
            "roots": [{"name": p.name or str(p), "path": str(p)} for p in picker_roots()],
            "dirs": list_subdirs(path),
            "images_here": len(rep.files),
            "images_recursive": len(deep.files),
            "note": deep.explain_empty() if not deep.files else "",
        })

    # ---------------------------------------------------------------- indexing

    @app.post("/api/pick-folder")
    def api_pick_folder():
        """Open the OS folder chooser on the machine running this server."""
        from . import native_dialog

        # The dialog appears on the server's desktop, so it only makes sense when
        # the browser and the server are the same machine.
        if request.remote_addr not in ("127.0.0.1", "::1", "localhost"):
            return jsonify({"error": "the native dialog is only available when the "
                                     "browser and server are on the same machine"}), 400
        ok, why = native_dialog.available()
        if not ok:
            return jsonify({"error": why}), 501

        body = request.get_json(force=True, silent=True) or {}
        try:
            chosen = native_dialog.choose_folder(body.get("start") or None)
        except native_dialog.DialogUnavailable as e:
            return jsonify({"error": str(e)}), 500
        if not chosen:
            return jsonify({"cancelled": True})

        path = Path(chosen)
        if not path.is_dir():
            return jsonify({"error": f"{chosen} is not a directory"}), 400
        if not within_picker_roots(path):
            return jsonify({"error": f"{chosen} is outside your home directory "
                                     f"and mounted volumes"}), 403
        deep = scan([path], recursive=True)
        return jsonify({
            "path": str(path),
            "images_recursive": len(deep.files),
            "note": deep.explain_empty() if not deep.files else "",
        })

    @app.get("/api/pick-folder/available")
    def api_pick_folder_available():
        from . import native_dialog

        ok, why = native_dialog.available()
        local = request.remote_addr in ("127.0.0.1", "::1", "localhost")
        return jsonify({"available": ok and local,
                        "backend": native_dialog.backend(),
                        "reason": "" if (ok and local) else
                                  (why if not ok else "browser is not on this machine")})

    @app.post("/api/index")
    def api_index():
        body = request.get_json(force=True, silent=True) or {}
        folders = [f for f in (body.get("folders") or []) if f]
        if not folders:
            return jsonify({"error": "no folders given"}), 400
        for f in folders:
            if not within_picker_roots(Path(f)):
                return jsonify({"error": f"{f} is outside your home directory and mounted volumes"}), 403

        # A run the user started takes priority: ask the watcher to stand aside
        # and wait for it to release the lock before claiming it ourselves.
        if watcher.active:
            if not watcher.yield_to_user(timeout=30):
                watcher.resume()
                return jsonify({"error": "a background folder check is still finishing; "
                                         "try again in a moment"}), 409

        if job.is_running():
            watcher.resume()
            return jsonify({"error": "an indexing run is already in progress",
                            "indexing": job.snapshot()}), 409

        with job.lock:
            if job.is_running():
                watcher.resume()
                return jsonify({"error": "an indexing run is already in progress"}), 409
            job.reset()
            job.running = True
            job.folders = folders
            job.started = time.time()

        def progress(ev: dict):
            kind = ev.get("event")
            if kind == "start":
                job.total = ev.get("total", 0)
            elif kind in ("photo", "fail", "skip"):
                job.done = ev.get("n", job.done)
                job.current = Path(ev.get("path", "")).name
                if kind != "skip":
                    job.log.append({
                        "path": ev.get("path"), "name": Path(ev.get("path", "")).name,
                        "caption": ev.get("caption", ""),
                        "status": ev.get("caption_status", "fail" if kind == "fail" else ""),
                        "error": ev.get("error", ""),
                    })
            elif kind == "warn":
                job.notes.append(ev.get("message", ""))

        def run():
            try:
                stats = index_folders(
                    [Path(f) for f in folders],
                    vision_model=body.get("vision_model") or settings.get("vision_model") or None,
                    caption=bool(body.get("caption", settings.get("caption_enabled", True))),
                    do_ocr=bool(body.get("ocr", settings.get("ocr_enabled", True))),
                    alt_ocr=body.get("alt_ocr") or settings.get("alt_ocr", "auto"),
                    recursive=bool(body.get("recursive", True)),
                    reindex=bool(body.get("reindex", False)),
                    limit=int(body["limit"]) if body.get("limit") else None,
                    device=device,
                    progress=progress,
                    store=store,
                    should_stop=job.cancel.is_set,
                )
                job.summary = stats.as_dict()
                job.notes = list(dict.fromkeys(job.notes + stats.notes))
            except IndexBusy as e:
                job.error = str(e)
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.running = False
                watcher.resume()          # background checks may continue
                refresh_served()

        job.thread = threading.Thread(target=run, daemon=True)
        job.thread.start()
        return jsonify({"started": True})

    @app.get("/api/storage")
    def api_storage():
        from . import storage

        return jsonify(storage.report())

    @app.post("/api/storage/location")
    def api_storage_location():
        """Choose where the index lives from now on. Takes effect on restart."""
        from . import storage

        body = request.get_json(force=True, silent=True) or {}
        if body.get("reset"):
            storage.clear_location()
            return jsonify({"ok": True, "restart_required": True,
                            "message": "reverted to the default location; "
                                       "restart Ophotofinder to use it"})
        raw = (body.get("path") or "").strip()
        if not raw:
            return jsonify({"error": "no path given"}), 400
        target = Path(raw).expanduser()
        if not within_picker_roots(target):
            return jsonify({"error": f"{raw} is outside your home directory "
                                     f"and mounted volumes"}), 403
        try:
            chosen = storage.set_location(str(target))
        except (OSError, PermissionError) as e:
            return jsonify({"error": str(e)}), 400

        from .config import DATA_DIR
        return jsonify({
            "ok": True, "path": str(chosen), "restart_required": True,
            "previous": str(DATA_DIR),
            "message": (f"New photos will be indexed at {chosen} after a restart. "
                        f"The existing index is still at {DATA_DIR} — move it with: "
                        f"ophotofinder storage --move-to \"{chosen}\""),
        })

    @app.post("/api/index/clear")
    def api_index_clear():
        """Empty the index. Photo files are never touched."""
        from .config import THUMB_DIR as _THUMBS

        if job.is_running() or read_holder():
            return jsonify({"error": "an indexing run is in progress; "
                                     "wait for it to finish"}), 409

        before = store.counts()
        ids = [i for i, _ in store.records()]
        store.drop_all()
        removed = 0
        for i in ids:
            t = _THUMBS / f"{i}.jpg"
            if t.exists():
                t.unlink(missing_ok=True)
                removed += 1
        refresh_served()
        return jsonify({"ok": True, "removed": before.get("clip", 0),
                        "thumbnails_removed": removed, "counts": store.counts()})

    @app.post("/api/ollama/test")
    def api_ollama_test():
        """Check an Ollama address without saving it."""
        from .config import normalise_host

        body = request.get_json(force=True, silent=True) or {}
        candidate = normalise_host(body.get("host", "")) or ollama.host()
        try:
            r = requests.get(f"{candidate}/api/tags", timeout=5)
            r.raise_for_status()
            names = sorted(m["name"] for m in r.json().get("models", []))
            return jsonify({"ok": True, "host": candidate, "models": len(names),
                            "sample": names[:6]})
        except Exception as e:
            return jsonify({"ok": False, "host": candidate,
                            "error": f"{type(e).__name__}: {e}"})

    @app.get("/api/settings")
    def api_settings_get():
        s = settings.load(refresh=True)
        try:
            models = [
                {"name": m.name, "vision": m.has_vision, "broken": m.is_broken,
                 "ctx": m.context_length, "caps": m.capabilities,
                 "size": m.size, "fit": m.fits()}
                for m in ollama.inventory()
            ]
        except Exception:
            models = []
        auto_v = ""
        try:
            auto_v = ollama.pick_vision_model().name
        except Exception:
            pass
        auto_t = ollama.pick_text_model()
        from . import sysinfo

        return jsonify({
            "settings": s,
            "machine": sysinfo.summary(),
            "loaded": sysinfo.running_offload(),
            "models": models,
            "auto_vision": auto_v,
            "auto_text": auto_t.name if auto_t else "",
            "watcher": watcher.status(),
        })

    @app.post("/api/settings")
    def api_settings_set():
        body = request.get_json(force=True, silent=True) or {}
        body.pop("folders", None)          # folders have their own endpoints
        before = settings.load()
        saved = settings.save(body)
        # Only nudge the watcher when something it cares about actually changed --
        # saving an unrelated preference must not start indexing.
        if before.get("ollama_host") != saved.get("ollama_host"):
            ollama.clear_cache()
        if any(before.get(k) != saved.get(k)
               for k in ("watch_enabled", "watch_interval_minutes")):
            watcher.poke()
        return jsonify({"settings": saved, "watcher": watcher.status()})

    @app.post("/api/settings/folders")
    def api_settings_folders():
        """Add, remove, or toggle recursion on a watched folder."""
        body = request.get_json(force=True, silent=True) or {}
        action = body.get("action")
        raw = body.get("path") or ""
        if not raw:
            return jsonify({"error": "no path given"}), 400
        path = Path(raw).expanduser()
        if action in ("add", "recursive") and not within_picker_roots(path):
            return jsonify({"error": f"{raw} is outside your home directory "
                                     f"and mounted volumes"}), 403

        if action == "add":
            if not path.is_dir():
                return jsonify({"error": f"{raw} is not a folder"}), 400
            saved = settings.add_folder(str(path), bool(body.get("recursive", True)))
        elif action == "remove":
            saved = settings.remove_folder(str(path))
        elif action == "recursive":
            saved = settings.set_folder_recursive(str(path), bool(body.get("recursive", True)))
        else:
            return jsonify({"error": f"unknown action {action!r}"}), 400

        watcher.poke()
        out = []
        for f in saved["folders"]:
            fp = Path(f["path"])
            out.append({**f, "exists": fp.is_dir(),
                        "images": len(scan([fp], recursive=f.get("recursive", True)).files)
                                  if fp.is_dir() else 0})
        return jsonify({"folders": out, "watcher": watcher.status()})

    @app.get("/api/settings/folders")
    def api_settings_folders_get():
        out = []
        for f in settings.folders():
            fp = Path(f["path"])
            out.append({**f, "exists": fp.is_dir(),
                        "images": len(scan([fp], recursive=f.get("recursive", True)).files)
                                  if fp.is_dir() else 0})
        return jsonify({"folders": out, "watcher": watcher.status()})

    @app.post("/api/watch/run")
    def api_watch_run():
        """Check the watched folders right now rather than waiting for the timer."""
        if job.is_running():
            return jsonify({"error": "an indexing run is already in progress"}), 409
        watcher.poke(force=True)
        return jsonify({"poked": True, "watcher": watcher.status()})

    @app.get("/api/pipeline")
    def api_pipeline():
        """Current pipeline versions and what in the index is behind them."""
        from .config import COMPONENT_COST, PIPELINE_VERSIONS
        from .indexer import plan_upgrade

        plan = plan_upgrade(store)
        return jsonify({
            **plan,
            "components": [
                {"name": k, "version": v,
                 "cost": COMPONENT_COST.get(k, "?"),
                 "behind": plan["per_component"].get(k, 0)}
                for k, v in sorted(PIPELINE_VERSIONS.items())
            ],
        })

    @app.get("/api/system")
    def api_system():
        """What this machine can do -- the doctor report, for the settings tab."""
        import platform

        from .clip_embed import pick_device
        from .config import DATA_DIR
        from .imaging import HEIF_OK
        from .ocr import TESSERACT_OK
        from .ocr_alt import best_available, describe_available

        readers = [{"name": n, "ok": ok, "note": note} for n, ok, note in describe_available()]
        return jsonify({
            "platform": platform.platform(),
            "python": platform.python_version(),
            "data_dir": str(DATA_DIR),
            "device": pick_device(),
            "heic": HEIF_OK,
            "tesseract": TESSERACT_OK,
            "second_readers": readers,
            "second_reader_default": best_available(),
            "ollama_host": ollama.host(),
            "counts": store.counts(),
            "caption_health": store.caption_health(),
            "folders": store.folders(),
        })

    @app.post("/api/upgrade")
    def api_upgrade():
        """Re-run only the components that are behind, as a background job."""
        from .indexer import upgrade_index

        body = request.get_json(force=True, silent=True) or {}
        # A run the user started takes priority: ask the watcher to stand aside
        # and wait for it to release the lock before claiming it ourselves.
        if watcher.active:
            if not watcher.yield_to_user(timeout=30):
                watcher.resume()
                return jsonify({"error": "a background folder check is still finishing; "
                                         "try again in a moment"}), 409

        if job.is_running():
            watcher.resume()
            return jsonify({"error": "an indexing run is already in progress",
                            "indexing": job.snapshot()}), 409
        with job.lock:
            if job.is_running():
                watcher.resume()
                return jsonify({"error": "an indexing run is already in progress"}), 409
            job.reset()
            job.running = True
            job.mode = "upgrade"
            job.folders = ["(upgrading existing records)"]
            job.started = time.time()

        only = body.get("only") or None

        def progress(ev: dict):
            kind = ev.get("event")
            if kind == "start":
                job.total = ev.get("total", 0)
            elif kind in ("photo", "fail"):
                job.done = ev.get("n", job.done)
                job.current = Path(ev.get("path", "")).name
                job.log.append({
                    "path": ev.get("path"), "name": Path(ev.get("path", "")).name,
                    "caption": ev.get("upgraded", ""),
                    "status": "fail" if kind == "fail" else "ok",
                    "error": ev.get("error", ""),
                })

        def run():
            try:
                stats = upgrade_index(
                    vision_model=body.get("vision_model") or None,
                    alt_ocr=body.get("alt_ocr", "auto"),
                    device=device, limit=int(body["limit"]) if body.get("limit") else None,
                    only=only, progress=progress, store=store,
                    should_stop=job.cancel.is_set,
                )
                job.summary = stats.as_dict()
                job.notes = list(dict.fromkeys(job.notes + stats.notes))
            except IndexBusy as e:
                job.error = str(e)
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.running = False
                job.mode = "index"
                watcher.resume()
                refresh_served()

        job.thread = threading.Thread(target=run, daemon=True)
        job.thread.start()
        return jsonify({"started": True})

    @app.get("/api/index/status")
    def api_index_status():
        return jsonify(job.snapshot())

    return app
