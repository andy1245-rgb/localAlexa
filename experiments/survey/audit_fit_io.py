"""Every file, temp, mkdir, rmdir and list call one fit makes, per fit, in one process.

    .venv-embed/bin/python experiments/survey/audit_fit_io.py --fits 2

Why this exists: the first full sweep on the Windows box died with
``OSError: [Errno 22] Invalid argument`` and the traceback was thrown away, so
the raising call was never identified. ``sys.addaudithook`` is cross-platform
and records the OS operations CPython itself performs, which makes the question
"what does fit N touch that fit 1 did not" answerable on either platform — and
on the Windows box, where the platform's own error text is the only evidence,
it is the only way to see the shape of the write/delete cycle without a profiler.

The interesting number is not the event count but the *delta*: what the second
and later fits do that the first did not. Reading it is what showed that the
only filesystem cycle the full sweep performs and the smoke run does not is the
trainer's own output tree in the temp directory — one create, one ~260 MB write,
one list, one recursive delete per fit.

This is a diagnostic, not part of the sweep. It fits the real recipe on the real
pools, so it costs one fine-tune per fit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_env import REPO_ROOT
from survey_setfit import SurveySetFitPredictor

WATCHED = {
    "open",
    "os.listdir",
    "os.scandir",
    "os.stat",
    "os.mkdir",
    "os.rmdir",
    "os.remove",
    "os.rename",
    "os.replace",
    "os.chmod",
    "os.utime",
    "os.symlink",
    "os.link",
    "os.truncate",
    "os.fsync",
    "tempfile.mkstemp",
    "mmap.__new__",
    "shutil.copyfile",
    "shutil.move",
    "subprocess.Popen",
    "socket.connect",
    "os.chdir",
}


def _shorten(value: object, limit: int = 150) -> str:
    text = str(value).replace(str(Path.home()), "~").replace(str(REPO_ROOT), "<REPO>")
    return text if len(text) <= limit else text[: limit // 2] + "..." + text[-(limit // 2) :]


class Audit:
    def __init__(self) -> None:
        self.phase = "start"
        self.events: list[tuple[str, str, str]] = []

    def hook(self, event: str, args: tuple) -> None:
        if event in WATCHED:
            self.events.append((self.phase, event, _shorten(args[0]) if args else ""))

    def counts(self) -> dict[str, int]:
        return {
            f"{phase}|{event}": count
            for (phase, event), count in sorted(Counter((p, e) for p, e, _ in self.events).items())
        }

    def delta(self, first: str, later: str) -> list[tuple[str, str]]:
        before = {(e, d) for phase, e, d in self.events if phase == first}
        after = {(e, d) for phase, e, d in self.events if phase == later}
        return sorted(after - before)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Record the OS operations each SetFit fit performs, and diff fit 1 against fit 2."
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--fits", type=int, default=2, help="how many fits to run (default 2)")
    parser.add_argument("--out", help="optional path to write the event dump as JSON")
    args = parser.parse_args(argv)

    from localalexa.ladder.pools import fold_pool, reference_pool
    from localalexa.ladder.run import load_bench

    bench = load_bench()
    indices = bench.folds.indices(list(bench.families))[0]
    held_out = {bench.families[i] for i in indices}
    first_pool = fold_pool(bench.registry, held_out)
    all_pool = reference_pool(bench.registry)
    texts_for = {0: [t for t, _ in first_pool], 1: [t for t, _ in all_pool]}
    labels_for = {0: [l for _, l in first_pool], 1: [l for _, l in all_pool]}

    audit = Audit()
    sys.addaudithook(audit.hook)
    for fit in range(max(args.fits, 1)):
        pool = 0 if fit == 0 else 1
        audit.phase = f"fit{fit + 1}"
        started = time.perf_counter()
        predictor = SurveySetFitPredictor(args.model).fit(texts_for[pool], labels_for[pool])
        elapsed = time.perf_counter() - started
        audit.phase = f"score{fit + 1}"
        predictor.scores(bench.texts[:32])
        print(f"fit {fit + 1}: {elapsed:.1f}s on {len(texts_for[pool])} references", flush=True)

    counts = audit.counts()
    print(json.dumps(counts, indent=2))
    if args.fits > 1:
        delta = audit.delta("fit1", "fit2")
        print(f"fit2-only (event, path) pairs: {len(delta)}")
        for event, detail in delta:
            print(f"  {event}  {detail}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "fits": args.fits,
                    "counts": counts,
                    "events": [
                        {"phase": p, "event": e, "detail": d} for p, e, d in audit.events
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
