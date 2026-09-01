"""Ophotofinder command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ollama, settings
from .config import CHROMA_DIR, DATA_DIR
from .imaging import HEIF_OK
from .indexer import index_folders
from .lock import IndexBusy
from .ocr import TESSERACT_OK
from .scan import scan
from .search import Searcher, answer, build_where
from .store import Store

BOLD, DIM, GREEN, YELLOW, RED, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m")


def _c(s, colour):
    return f"{colour}{s}{RESET}" if sys.stdout.isatty() else str(s)


# ------------------------------------------------------------------ commands

def cmd_models(args) -> int:
    try:
        inv = ollama.inventory()
    except ollama.OllamaError as e:
        print(_c(str(e), RED))
        return 2
    if not inv:
        print("No models installed. Try: ollama pull moondream")
        return 1

    print(_c(f"Ollama at {ollama.host()}", BOLD))
    from . import sysinfo
    accel = sysinfo.detect_accelerator()
    print(_c(f"this machine: {accel['summary']}", DIM))
    print(f"\n{'MODEL':<26} {'RUNTIME':<16} {'SIZE':>7} {'MAX CTX':>9}  SUITABILITY")
    for m in inv:
        note = ""
        if m.is_broken:
            note = _c("broken on this Ollama (mllama)", RED)
        elif m.has_vision:
            note = _c("usable for captioning", GREEN) + (
                f" {DIM}(small: short prompt){RESET}" if m.is_small else "")
        fit = m.fits(accel)
        colour = {"good": GREEN, "ok": DIM, "warn": YELLOW, "bad": RED}.get(fit["level"], DIM)
        verdict = _c(fit["headline"], colour)
        if note:
            verdict = note + "; " + verdict
        size = f"{m.size/1e9:.1f}GB" if m.size else "?"
        print(f"{m.name:<26} {fit['runtime']['label']:<16} {size:>7} "
              f"{m.context_length or '?':>9}  {verdict}")
        for reason in fit["reasons"][:2]:
            print(_c(f"{'':>26}   - {reason}", DIM))

    try:
        v = ollama.pick_vision_model()
        print(f"\nauto vision  -> {_c(v.name, CYAN)}  (num_ctx {v.context_length or 'default'})")
    except ollama.OllamaError as e:
        print(f"\nauto vision  -> {_c('none: ' + str(e), YELLOW)}")
    t = ollama.pick_text_model()
    print(f"auto text    -> {_c(t.name if t else 'none', CYAN)}")
    return 0


def cmd_doctor(args) -> int:
    print(_c("Ophotofinder doctor", BOLD))
    print(f"  data dir       {DATA_DIR}")
    print(f"  chroma         {CHROMA_DIR}")
    print(f"  HEIC support   {_c('yes', GREEN) if HEIF_OK else _c('NO - install pillow-heif', RED)}")
    print(f"  tesseract OCR  {_c('yes', GREEN) if TESSERACT_OK else _c('no - OCR will be skipped', YELLOW)}")
    if TESSERACT_OK:
        import subprocess
        try:
            langs = [l for l in subprocess.run(["tesseract", "--list-langs"],
                     capture_output=True, text=True, timeout=5).stdout.splitlines()[1:] if l.strip()]
        except Exception:
            langs = []
        real = [l for l in langs if l not in ("osd", "snum")]   # not languages
        note = "" if len(real) > 1 else _c(
            "   <- English only; `brew install tesseract-lang` adds ~100 more", YELLOW)
        print(f"    languages    {', '.join(langs) or '?'}{note}")
    from .ocr_alt import apple_languages, best_available, describe_available
    chosen = best_available()
    print(f"  2nd reader     {_c(chosen, GREEN) if chosen != 'none' else _c('none available', YELLOW)}")
    for name, ok, note in describe_available():
        mark = _c('yes', GREEN) if ok else _c('no ', DIM)
        extra = f" ({len(apple_languages())} languages)" if (ok and name == 'apple') else ""
        print(f"    {name:10} {mark}  {note}{extra}")
    if chosen == "none":
        print(_c("    install one: pip install 'ophotofinder[ocr]'  (any platform)", YELLOW))
    up = ollama.is_up()
    print(f"  ollama         {_c('up at ' + ollama.host(), GREEN) if up else _c('DOWN at ' + ollama.host(), RED)}")

    from . import sysinfo
    from .clip_embed import pick_device
    accel = sysinfo.detect_accelerator()
    print(f"  torch device   {pick_device()}")
    print(f"  accelerator    {accel['summary']}")
    loaded = sysinfo.running_offload()
    for l in loaded:
        print(f"    loaded now   {l['name']}: {l['note']}")

    if up:
        vm = ollama.vision_models()
        print(f"  vision models  {', '.join(m.name for m in vm) or _c('none (ollama pull moondream)', YELLOW)}")

    st = Store()
    c = st.counts()
    print(f"  indexed        {c['clip']} photos (clip) / {c['text']} (text)")
    health = st.caption_health()
    if health:
        print(f"  captions       " + ", ".join(f"{k}={v}" for k, v in sorted(health.items())))

    pics = Path.home() / "Pictures"
    if pics.is_dir():
        rep = scan([pics], recursive=True)
        if not rep.files:
            print(_c(f"\n  ~/Pictures: {rep.explain_empty()}", YELLOW))
        else:
            print(f"\n  ~/Pictures: {len(rep.files)} loose image files found")
    return 0


def cmd_index(args) -> int:
    roots = [Path(p).expanduser() for p in args.folders]
    missing = [r for r in roots if not r.exists()]
    if missing:
        print(_c(f"No such folder: {', '.join(str(m) for m in missing)}", RED))
        return 2

    if not args.no_caption:
        try:
            chosen = ollama.pick_vision_model(
                args.vision_model or settings.get("vision_model") or None)
            fit = chosen.fits()
            if fit["level"] == "bad":
                print(_c(f"WARNING: {fit['message']}", RED))
            elif fit["level"] == "warn":
                print(_c(f"Warning: {fit['message']}", YELLOW))
        except ollama.OllamaError:
            pass

    state = {"last": 0}

    def progress(ev: dict):
        kind = ev.get("event")
        if kind == "start":
            print(f"{ev['total']} image file(s) found\n")
        elif kind == "photo":
            n, total = ev["n"], ev["total"]
            status = ev["caption_status"]
            mark = {"ok": _c("cap", GREEN), "empty": _c("cap!", YELLOW),
                    "error": _c("cap!", RED)}.get(status, _c("cap-", DIM))
            ocr = _c("ocr", CYAN) if ev.get("ocr") else _c("---", DIM)
            alt = _c("alt", GREEN) if ev.get("ocr_alt") else _c("---", DIM)
            name = Path(ev["path"]).name
            line = f"[{n}/{total}] {mark} {ocr} {alt}  {name[:40]:<40}"
            cap = (ev.get("caption") or "").replace("\n", " ")
            print(line + _c(cap[:60], DIM))
        elif kind == "skip":
            pass
        elif kind == "fail":
            print(f"[{ev['n']}/{ev['total']}] {_c('FAIL', RED)} {Path(ev['path']).name}: {ev['error'][:80]}")
        elif kind == "warn":
            print(_c("\n!! " + ev["message"] + "\n", YELLOW))

    try:
        stats = index_folders(
            roots,
            vision_model=args.vision_model or settings.get("vision_model") or None,
            caption=(not args.no_caption) and settings.get("caption_enabled", True),
            do_ocr=(not args.no_ocr) and settings.get("ocr_enabled", True),
            alt_ocr=args.alt_ocr if args.alt_ocr != "auto" else settings.get("alt_ocr", "auto"),
            alt_langs=[l.strip() for l in args.alt_langs.split(',')] if args.alt_langs else None,
            recursive=not args.no_recursive,
            reindex=args.reindex,
            limit=args.limit,
            device=args.device,
            progress=progress if not args.quiet else None,
        )
    except IndexBusy as e:
        print(_c(str(e), RED))
        return 3
    except ollama.OllamaError as e:
        print(_c(str(e), RED))
        return 2

    print()
    print(_c("--- indexing summary ---", BOLD))
    print(f"  found                {stats.found}")
    print(f"  indexed              {_c(stats.indexed, GREEN)}")
    print(f"  unchanged (skipped)  {stats.skipped_unchanged}")
    print(f"  failed               {_c(stats.failed, RED if stats.failed else DIM)}")
    print(f"  captions ok/empty/off {stats.caption_ok}/{stats.caption_empty}/{stats.caption_skipped}")
    print(f"  photos with OCR text {stats.ocr_hits} (tesseract) / {stats.alt_ocr_hits} (second reader)")
    print(f"  elapsed              {stats.elapsed:.1f}s")
    for n in stats.notes:
        print(f"  {_c('note:', CYAN)} {n}")
    for e in stats.errors[:10]:
        print(f"  {_c('error:', RED)} {e}")
    if len(stats.errors) > 10:
        print(f"  ... and {len(stats.errors) - 10} more errors")
    return 0 if stats.indexed or stats.skipped_unchanged else 1


def cmd_search(args) -> int:
    st = Store()
    if not st.counts()["clip"]:
        print(_c("Index is empty. Run: ophotofinder index <folder>", YELLOW))
        return 1

    where = build_where(args.year, args.camera, args.folder, args.with_text or None)
    s = Searcher(st, device=args.device)
    hits = s.search(" ".join(args.query), k=args.k, where=where,
                    use_clip=not args.text_only, use_text=not args.clip_only)

    if args.json:
        print(json.dumps([h.as_dict() for h in hits], indent=2))
        return 0

    if not hits:
        print("No matching photos.")
        return 1

    print(_c(f'\n{len(hits)} match(es) for "{" ".join(args.query)}"\n', BOLD))
    for i, h in enumerate(hits, 1):
        m = h.meta
        print(f"{_c(f'{i:>2}.', BOLD)} {h.path}")
        meta_bits = [b for b in (m.get("taken_at", "")[:10], m.get("camera"),
                                 m.get("orientation")) if b]
        print(f"    {_c(h.why + '  score ' + f'{h.score:.4f}', DIM)}"
              + (f"  {_c(' | '.join(meta_bits), DIM)}" if meta_bits else ""))
        cap = m.get("caption")
        if cap:
            print(f"    {cap[:180]}")
        elif m.get("caption_status") in ("empty", "error", "skipped"):
            why = f"{m.get('caption_status', 'unknown')}: {m.get('caption_detail', '')}".strip(": ")
            print(f"    {_c('no caption — ' + why, DIM)}")
        if m.get("ocr"):
            print(f"    {_c('OCR: ' + str(m['ocr'])[:120], CYAN)}")
        if m.get("ocr_alt") and str(m.get("ocr_alt")) != str(m.get("ocr", "")):
            label = f"OCR/{m.get('ocr_alt_method', 'alt')}: "
            print(f"    {_c(label + str(m['ocr_alt'])[:120].replace(chr(10), ' / '), GREEN)}")
        print()

    if args.answer:
        print(_c("--- answer ---", BOLD))
        try:
            print(answer(" ".join(args.query), hits,
                         model=args.text_model or settings.get("text_model") or None))
        except ollama.OllamaError as e:
            print(_c(str(e), RED))
    return 0


def cmd_status(args) -> int:
    st = Store()
    c = st.counts()
    print(_c("Ophotofinder index", BOLD))
    print(f"  photos      {c['clip']}")
    print(f"  text records {c['text']}")
    health = st.caption_health()
    if health:
        print("  captions   " + ", ".join(f"{k}={v}" for k, v in sorted(health.items())))
    from .config import pipeline_signature
    from .indexer import plan_upgrade
    plan = plan_upgrade(st)
    print(f"  pipeline   {pipeline_signature()}")
    if plan["stale"]:
        print(_c(f"  {plan['stale']} photo(s) behind the current pipeline "
                 f"- run: ophotofinder upgrade", YELLOW))
        for comp, n in sorted(plan["per_component"].items(), key=lambda kv: -kv[1]):
            print(_c(f"    {comp:10} {n:>6}", DIM))
    elif plan["total"]:
        print(_c("  all photos are at the current pipeline version", GREEN))

    folders = st.folders()
    if folders:
        print("\n  folders:")
        for f, n in list(folders.items())[:30]:
            print(f"    {n:>6}  {f}")
        if len(folders) > 30:
            print(f"    ... {len(folders) - 30} more")
    from . import server_pid
    from .lock import read_holder
    holder = read_holder()
    if holder:
        print(_c(f"\n  an indexing run is in progress (pid {holder['pid']})", YELLOW))
    web = server_pid.read()
    if web:
        print(_c(f"\n  web server running: pid {web['pid']} at "
                 f"http://{web['host']}:{web['port']} (since {web['started']})", CYAN))
        print(_c(f"  stop it with: ophotofinder web --stop", DIM))
    return 0


def cmd_clean(args) -> int:
    """Remove records from the index. Never touches the photo files themselves."""
    from .config import THUMB_DIR
    from .lock import read_holder

    holder = read_holder()
    if holder:
        print(_c(f"An indexing run is in progress (pid {holder['pid']}). "
                 f"Wait for it to finish before cleaning.", RED))
        return 3

    st = Store()
    records = st.records()
    if not records:
        print("The index is already empty.")
        return 0

    if args.folder:
        root = str(Path(args.folder).expanduser().resolve())
        targets = [(i, p) for i, p in records
                   if p == root or p.startswith(root + "/")]
        what = f"{len(targets)} photo(s) under {root}"
    elif args.missing:
        targets = [(i, p) for i, p in records if not p or not Path(p).exists()]
        what = f"{len(targets)} record(s) whose file no longer exists"
    else:
        targets = records
        what = f"the entire index ({len(targets)} photo(s))"

    if not targets:
        print(f"Nothing to clean: no records matched.")
        return 0

    print(f"This will remove {_c(what, BOLD)} from the index.")
    print(_c("Your photo files are not touched.", DIM))
    if not args.yes:
        for _, p in targets[:5]:
            print(f"    {p}")
        if len(targets) > 5:
            print(f"    ... and {len(targets) - 5} more")
        try:
            reply = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("Cancelled.")
            return 1

    ids = [i for i, _ in targets]
    if len(targets) == len(records):
        st.drop_all()          # cheaper and leaves no orphaned vectors
    else:
        st.delete(ids)

    removed_thumbs = 0
    for i in ids:
        t = THUMB_DIR / f"{i}.jpg"
        if t.exists():
            t.unlink(missing_ok=True)
            removed_thumbs += 1

    after = st.counts()
    print(_c(f"Removed {len(ids)} record(s) and {removed_thumbs} thumbnail(s).", GREEN))
    print(f"  index now holds {after['clip']} photo(s)")
    return 0


def cmd_upgrade(args) -> int:
    """Bring an existing index up to the current pipeline, component by component."""
    from .config import COMPONENT_COST, PIPELINE_VERSIONS
    from .indexer import plan_upgrade, upgrade_index

    plan = plan_upgrade()
    print(_c("pipeline " + plan["signature"], BOLD))
    print(f"  {plan['total']} photo(s) in the index")
    print(f"  {_c(plan['up_to_date'], GREEN)} up to date, "
          f"{_c(plan['stale'], YELLOW if plan['stale'] else DIM)} behind"
          + (f", {_c(plan['missing_file'], RED)} whose file is gone" if plan["missing_file"] else ""))

    if plan["per_component"]:
        print("\n  behind, by component:")
        for comp, n in sorted(plan["per_component"].items(), key=lambda kv: -kv[1]):
            print(f"    {comp:10} v{PIPELINE_VERSIONS[comp]}  {n:>6} photo(s)   "
                  f"{_c('(' + COMPONENT_COST.get(comp, '?') + ' to recompute)', DIM)}")

    if args.check:
        return 0
    if not plan["stale"]:
        print(_c("\nNothing to upgrade.", GREEN))
        return 0

    only = [c.strip() for c in args.only.split(",")] if args.only else None
    if only:
        print(f"\n  restricting to: {', '.join(only)}")
    if not args.yes:
        try:
            if input("\nRun the upgrade? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Cancelled.")
                return 1
        except EOFError:
            print("Cancelled.")
            return 1

    def progress(ev):
        if ev.get("event") == "photo":
            print(f"[{ev['n']}/{ev['total']}] {_c(ev.get('upgraded', ''), CYAN):<30} "
                  f"{Path(ev['path']).name[:44]}")
        elif ev.get("event") == "fail":
            print(f"[{ev['n']}/{ev['total']}] {_c('FAIL', RED)} {Path(ev['path']).name}: {ev['error'][:70]}")

    try:
        stats = upgrade_index(
            vision_model=args.vision_model, alt_ocr=args.alt_ocr,
            device=args.device, limit=args.limit, only=only,
            progress=None if args.quiet else progress,
        )
    except IndexBusy as e:
        print(_c(str(e), RED))
        return 3

    print()
    print(_c("--- upgrade summary ---", BOLD))
    print(f"  upgraded   {_c(stats.indexed, GREEN)}")
    print(f"  failed     {_c(stats.failed, RED if stats.failed else DIM)}")
    print(f"  elapsed    {stats.elapsed:.1f}s")
    for n in stats.notes:
        print(f"  {_c('note:', CYAN)} {n}")
    for e in stats.errors[:5]:
        print(f"  {_c('error:', RED)} {e}")
    return 0


def cmd_config(args) -> int:
    """Show or change the saved defaults."""
    if args.set:
        updates = {}
        for pair in args.set:
            if "=" not in pair:
                print(_c(f"expected key=value, got {pair!r}", RED))
                return 2
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            if k not in settings.DEFAULTS:
                print(_c(f"unknown setting {k!r}. Known: {', '.join(sorted(settings.DEFAULTS))}", RED))
                return 2
            default = settings.DEFAULTS[k]
            if isinstance(default, bool):
                v = v.lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int) and not isinstance(default, bool):
                try:
                    v = int(v)
                except ValueError:
                    print(_c(f"{k} expects a number", RED))
                    return 2
            elif isinstance(default, list):
                print(_c(f"{k} is managed with `config --add-folder` / `--remove-folder`", RED))
                return 2
            updates[k] = v
        settings.save(updates)
        print(_c("saved", GREEN))

    if args.add_folder:
        p = Path(args.add_folder).expanduser()
        if not p.is_dir():
            print(_c(f"{p} is not a folder", RED))
            return 2
        settings.add_folder(str(p), recursive=not args.no_recursive)
        print(_c(f"watching {p}" + ("" if not args.no_recursive else " (this folder only)"), GREEN))
    if args.remove_folder:
        settings.remove_folder(str(Path(args.remove_folder).expanduser()))
        print(_c(f"stopped watching {args.remove_folder}", GREEN))

    s = settings.load(refresh=True)
    print(_c("\nsaved settings", BOLD))
    for k in sorted(settings.DEFAULTS):
        if k == "folders":
            continue
        val = s.get(k)
        shown = _c("auto", DIM) if val == "" else str(val)
        print(f"  {k:24} {shown}")
    print(_c("\nwatched folders", BOLD))
    if not s["folders"]:
        print(_c("  none - add one with: ophotofinder config --add-folder ~/Photos", DIM))
    for f in s["folders"]:
        p = Path(f["path"])
        state = "" if p.is_dir() else _c("  (missing)", RED)
        scope = "with subfolders" if f.get("recursive", True) else "this folder only"
        print(f"  {f['path']}  {_c(scope, DIM)}{state}")
    return 0


def cmd_storage(args) -> int:
    """Show where the index lives and how big it is; optionally move it."""
    from . import storage
    from .lock import read_holder

    if args.move_to or args.set_location or args.reset_location:
        if read_holder():
            print(_c("An indexing run is in progress. Wait for it to finish.", RED))
            return 3
        try:
            if args.reset_location:
                storage.clear_location()
                print(_c("reverted to the default location "
                         "(takes effect on next start)", GREEN))
            elif args.move_to:
                dst, what = storage.move_data(args.move_to)
                print(_c(f"moved index to {dst} ({what})", GREEN))
            else:
                dst = storage.set_location(args.set_location)
                print(_c(f"new index location: {dst}", GREEN))
                print(_c("  the existing index was NOT moved; use --move-to to bring it", YELLOW))
        except (OSError, PermissionError, FileExistsError) as e:
            print(_c(str(e), RED))
            return 2
        print(_c("  restart Ophotofinder for this to take effect", DIM))
        return 0

    r = storage.report()
    print(_c("index storage", BOLD))
    print(f"  location   {r['path']}  {_c('(' + r['source'] + ')', DIM)}")
    print(f"  total      {_c(r['total_human'], BOLD)} in {r['total_files']} file(s)")
    for p in r["parts"]:
        print(f"    {p['label']:18} {p['human']:>10}  {p['files']} file(s)")
    print(f"  free space {r['free_human']}")
    print(_c("\n  move it with: ophotofinder storage --move-to /path/to/folder", DIM))
    return 0


def cmd_web(args) -> int:
    from . import server_pid
    from .web import create_app

    if args.stop:
        stopped, msg = server_pid.stop(port=args.port if args.port != 7777 else None)
        print(_c(msg, GREEN) if stopped else _c(msg, YELLOW))
        return 0 if stopped else 1

    running = server_pid.read()
    if running:
        print(_c(f"An Ophotofinder web server is already running: pid {running['pid']} "
                 f"at http://{running['host']}:{running['port']} "
                 f"(since {running['started']}).", YELLOW))
        print(f"Stop it with {_c('ophotofinder web --stop', BOLD)}, "
              f"or run this one on another port with --port.")
        return 1

    holder = server_pid.port_holder(args.port)
    if holder:
        print(_c(f"Port {args.port} is already in use by {holder}.", RED))
        print(f"That process was not started by Ophotofinder, so it is left alone. "
              f"Use --port to pick a free port.")
        return 1

    app = create_app(device=args.device)
    server_pid.write(args.host, args.port)
    print(_c(f"Ophotofinder web UI  ->  http://{args.host}:{args.port}", BOLD))
    print(_c(f"  stop it with: ophotofinder web --stop", DIM))
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        from .config import WEB_PID_PATH
        WEB_PID_PATH.unlink(missing_ok=True)
    return 0


# ------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ophotofinder",
        description="Local RAG over your photo folders. Ask in plain language, get photos back.",
    )
    p.add_argument("--device", default=None, help="mps | cuda | cpu (default: auto)")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("models", help="list installed Ollama models and their capabilities")
    m.set_defaults(func=cmd_models)

    d = sub.add_parser("doctor", help="check the local setup")
    d.set_defaults(func=cmd_doctor)

    i = sub.add_parser("index", help="index one or more photo folders")
    i.add_argument("folders", nargs="+")
    i.add_argument("--vision-model", default=None,
                   help="Ollama model for captions (must report the 'vision' capability)")
    i.add_argument("--no-caption", action="store_true", help="CLIP + OCR + EXIF only")
    i.add_argument("--no-ocr", action="store_true", help="skip Tesseract OCR")
    i.add_argument("--alt-ocr", default="auto",
                   choices=["auto", "apple", "rapidocr", "vlm", "none"],
                   help="second text reader (default: auto = best available)")
    i.add_argument("--alt-langs", default=None,
                   help="languages for the second reader, e.g. en-US,zh-Hant,ja-JP")
    i.add_argument("--no-recursive", action="store_true")
    i.add_argument("--reindex", action="store_true", help="re-process even unchanged files")
    i.add_argument("--limit", type=int, default=None)
    i.add_argument("--quiet", action="store_true")
    i.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="search the index")
    s.add_argument("query", nargs="+")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--year", default=None)
    s.add_argument("--camera", default=None)
    s.add_argument("--folder", default=None)
    s.add_argument("--with-text", action="store_true", help="only photos containing OCR text")
    s.add_argument("--clip-only", action="store_true")
    s.add_argument("--text-only", action="store_true")
    s.add_argument("--answer", action="store_true", help="also generate an answer with a local text model")
    s.add_argument("--text-model", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    st = sub.add_parser("status", help="what is in the index")
    st.set_defaults(func=cmd_status)

    c = sub.add_parser("clean", help="remove records from the index (photo files are never deleted)")
    g = c.add_mutually_exclusive_group()
    g.add_argument("--folder", default=None, help="only records under this folder")
    g.add_argument("--missing", action="store_true",
                   help="only records whose photo file no longer exists")
    c.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    c.set_defaults(func=cmd_clean)

    u = sub.add_parser("upgrade", help="bring the index up to the current pipeline version")
    u.add_argument("--check", action="store_true", help="report what is behind and exit")
    u.add_argument("--only", default=None,
                   help="upgrade only these components, e.g. ocr_alt,exif")
    u.add_argument("--vision-model", default=None)
    u.add_argument("--alt-ocr", default="auto",
                   choices=["auto", "apple", "rapidocr", "vlm", "none"])
    u.add_argument("--limit", type=int, default=None)
    u.add_argument("--yes", "-y", action="store_true")
    u.add_argument("--quiet", action="store_true")
    u.set_defaults(func=cmd_upgrade)

    cf = sub.add_parser("config", help="show or change saved defaults and watched folders")
    cf.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="e.g. --set vision_model=moondream --set watch_enabled=true")
    cf.add_argument("--add-folder", default=None, help="watch a folder for changes")
    cf.add_argument("--remove-folder", default=None)
    cf.add_argument("--no-recursive", action="store_true",
                    help="with --add-folder: do not include subfolders")
    cf.set_defaults(func=cmd_config)

    stg = sub.add_parser("storage", help="where the index lives and how big it is")
    stg.add_argument("--move-to", default=None, metavar="DIR",
                     help="move the index to DIR and use it from now on")
    stg.add_argument("--set-location", default=None, metavar="DIR",
                     help="use DIR from now on without moving existing data")
    stg.add_argument("--reset-location", action="store_true",
                     help="go back to ~/.ophotofinder")
    stg.set_defaults(func=cmd_storage)

    w = sub.add_parser("web", help="run the web UI")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=7777)
    w.add_argument("--stop", action="store_true",
                   help="stop a web server started earlier and exit")
    w.set_defaults(func=cmd_web)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
