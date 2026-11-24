# Phase 3 environment — the measurement workspace on `probe-box`

2026-11-23, branch `survey/base-model-2026-11-23`. Operator's goal for this stage: a working
**CPU** measurement workspace on the box, proved by one remote command that downloads a Hugging
Face sentence-transformer, loads it, embeds rows, and prints a JSON line.

Verdict: **the workspace works, and the download leg is proved — but not by the model the brief
named.** `all-MiniLM-L6-v2` was already sitting in this box's HF cache from 2026-09-21, so loading
it only exercised Hub *metadata*, not a weight download. A second model, provably absent from the
cache, was pulled from the Hub to close that gap (`07_download_proof.py`). Details in (f).

---

## (a) Remote path

```
~/base-model-survey                       workspace root (303 files / 28.3 MB excl. venv, +1,774 venv files)
~/base-model-survey/.venv                 venv, --system-site-packages
~/localalexa-subset.tgz                   the 3.93 MB tar that delivered the repo subset
~/0[1-9]_*.ps1                            the setup scripts, as run (archive copy: base-model-survey\setup\
                                           holds 01-06 and 08; 09_counts.ps1 postdates it)
```

Repo subset in the workspace: `data/ src/ experiments/ tests/ docs/ pyproject.toml results/ladder/`,
plus `probe.py`, `07_download_proof.py`, `setup/`, `pip-install.log`. No `.git`, no `checkpoints/`,
no `.venv*`, no `results/` outside `results/ladder/`.

Delivery was **tar over ssh**, never the sshfs mount:

```bash
tar -czf /tmp/localalexa-subset.tgz \
    --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' \
    -C ~/localAlexa \
    data src experiments tests docs pyproject.toml results/ladder
scp -o ConnectTimeout=8 /tmp/localalexa-subset.tgz probe-box:~/
ssh probe-box 'powershell -NoProfile -ExecutionPolicy Bypass -File ~/02_extract.ps1'
```

`02_extract.ps1` runs `tar.exe -xzf ~/localalexa-subset.tgz -C ~/base-model-survey`.

Verification (02_extract.ps1 output):

```
archive         = ~/localalexa-subset.tgz bytes=3932831
extract         = ok
remote_files    = 299
present data\routing_382.json = 41176 bytes
present results\ladder\rung5-ensemble.json = 36197 bytes
present pyproject.toml = 970 bytes
unexpected .git = 0
unexpected checkpoints = 0
unexpected .venv = 0
unexpected __pycache__ = 0
remote_total_mb = 28.31
```

299 remote files vs 299 entries in the tar, lists identical after normalising separators (`comm -3`
empty). Byte sizes of `data/routing_382.json` (41176) and `results/ladder/rung5-ensemble.json`
(36197) match the local files exactly.

Those 299 included 6 files of `src/localalexa.egg-info`, which was removed afterwards (finding 9), so
the repo subset now standing in the workspace is **293 files / 28.3 MB**. Full breakdown of the
workspace, re-measured after cleanup (`09_counts.ps1`):

```
repo_subset_files   = 293  (28.3 MB)
root_py_files       = 2  [07_download_proof.py probe.py]
setup_ps1_files     = 7
other_files         = 1  [pip-install.log]
total_excl_venv     = 303
venv_files          = 1774
check data\routing_382.json present=True bytes=41176
check results\ladder\rung5-ensemble.json present=True bytes=36197
check pyproject.toml present=True bytes=970
```

---

## (b) Exact command lines

Native interpreter, untouched:

```
~/AppData/Local/Programs/Python/Python311/python.exe   (3.11.9, pip 25.3)
```

Venv creation — `--system-site-packages` is what avoids re-downloading torch:

```
powershell> & ~/AppData/Local/Programs/Python/Python311/python.exe -m venv --system-site-packages ~/base-model-survey/.venv
```

Package install — dry-run first, because pip was about to be pointed at a venv that inherits the
box's whole system site, and any resolver decision that touched `torch` or `transformers` had to be
caught before it happened:

```
powershell> & ~/base-model-survey/.venv/Scripts/python.exe -m pip install --dry-run setfit
   ... Would install evaluate-0.4.6 fsspec-2025.3.0 requests-2.34.2 setfit-1.2.0
powershell> & ~/base-model-survey/.venv/Scripts/python.exe -m pip install setfit
   ... Successfully installed evaluate-0.4.6 fsspec-2025.3.0 requests-2.34.2 setfit-1.2.0
```

`setfit` was the **only** package the brief listed that was missing. `sentence-transformers`,
`scikit-learn`, `pandas` and `onnxruntime` were already installed in the native interpreter and are
inherited through `--system-site-packages`, so nothing was installed for them and no wheel was
downloaded twice. `pip freeze --local` confirms four distributions were added and nothing else:

```
evaluate==0.4.6
fsspec==2025.3.0
requests==2.34.2
setfit==1.2.0
```

`torch` and `transformers` were verified as inherited, not reinstalled — the dry run's plan contained
neither, and `Requirement already satisfied ... torch>=1.11.0` appears in the resolver output.

---

## (c) Version table

Read from the venv itself (`importlib.metadata` + `__version__`), on 2026-11-23.

| component | version | source |
|---|---|---|
| Python | 3.11.9 (CPython, AMD64, 64-bit) | native interpreter |
| venv | `~/base-model-survey/.venv` | `--system-site-packages` |
| torch | **2.11.0+cpu** (`torch_cuda=false`, `torch.version.cuda is None`) | inherited |
| transformers | **5.6.2** | inherited |
| sentence-transformers | 5.4.1 | inherited |
| setfit | **1.2.0** | installed into venv |
| scikit-learn | 1.8.0 | inherited |
| pandas | 2.3.3 | inherited |
| numpy | 2.4.4 | inherited |
| scipy | 1.17.1 | inherited |
| huggingface-hub | 1.12.0 | inherited |
| tokenizers | 0.22.2 | inherited |
| onnxruntime | 1.20.1 | inherited (see f.2) |
| evaluate / fsspec / requests | 0.4.6 / 2025.3.0 / 2.34.2 | installed into venv |
| pip (venv) | 24.0 | from `ensurepip` (see f.3) |
| pip (system) | 25.3 | native interpreter |

Hardware, from the box: Intel Core Ultra 7 258V, **8 logical cores**, **31.57 GB RAM**,
211.4 GB free on `C:`.

---

## (d) Probe output

`probe.py` writes one JSON line to stdout; everything human-readable goes to stderr, so a later stage
can pipe stdout straight into a parser. Canonical run, verbatim:

```
python                3.11.9 (CPython, AMD64)
executable            ~/base-model-survey/.venv/Scripts/python.exe
cores / ram           8 / 31.57 GB
torch                 2.11.0
transformers          5.6.2
sentence_transformers 5.4.1
setfit                1.2.0
scikit_learn          1.8.0
numpy                 2.4.4
huggingface_hub       1.12.0
torch threads / cuda  8 / False
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]Loading weights: 100%|##########| 103/103 [00:00<00:00, 4938.31it/s]
load sentence-transformers/all-MiniLM-L6-v2  2572 ms, dim=384
encodes               160 single-sentence calls (32 sentences x 5 passes)
latency per sentence  median 12.478 ms, p95 16.873 ms, first warm call 29 ms
snapshot 1110a243fdf4  87.3 MB total, 87.3 MB excluding ['1_Pooling']
hf repo cache dir     1019.0 MB at ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
{"arch":"AMD64","bits":64,"core_count":8,"embed_dim":384,"encodes":160,"encoding":"utf-8","executable":"~/base-model-survey/.venv/Scripts/python.exe","first_call_ms":28.999,"huggingface_hub":"1.12.0","load_ms":2572.3,"median_ms_per_sentence":12.4782,"model":"sentence-transformers/all-MiniLM-L6-v2","model_size_bytes":91578415,"numpy":"2.4.4","ok":true,"p95_ms_per_sentence":16.8727,"passes":5,"python":"3.11.9","python_impl":"CPython","ram_gb":31.57,"rss_mb":550.2,"scikit_learn":"1.8.0","sentences":32,"sentence_transformers":"5.4.1","setfit":"1.2.0","snapshot_bytes":91578415,"snapshot_extra_dirs":["1_Pooling"],"torch":"2.11.0","torch_cuda":false,"torch_threads":8,"transformers":"5.6.2"}
```

Three runs of the same script, for spread rather than a point estimate:

| run | load ms | median ms/sentence | p95 ms | first warm call ms |
|---|---:|---:|---:|---:|
| 1 (first, cold process) | 1875 | 13.805 | 20.010 | 109 |
| 2 | 1797 | 12.124 | 17.442 | 22 |
| 3 | 2572 | 12.478 | 16.873 | 29 |

**These are smoke-test numbers, not a benchmark.** Each call encodes exactly one sentence, batch
size 1, and the box was not quiesced. In particular, ~12.5 ms here must **not** be read against the
ladder's 7.75 ms SetFit figure (`docs/ladder-results-2026-11-10.md`) — different batching, different
process, no idle-box protocol. Phase 3's real latency numbers need the 382 rows, batch-size
declarations, and the same protocol the ladder used.

Embedding dimension: **384**, as expected for MiniLM-L6. Resident memory after load: 550 MB. Model on
disk: **87.3 MB** (the resolved snapshot), 1019 MB for the whole hub directory — see f.7.

### The download leg, separately proved

`07_download_proof.py` first asserts the target is **not** in the cache (it listed 14 repos present,
this one absent), then downloads and loads it:

```
hub cache            ~/.cache/huggingface/hub
repos present before 14: Cactus-Compute/needle3, NeoQuasar/Kronos-Tokenizer-2k, NeoQuasar/Kronos-Tokenizer-base, NeoQuasar/Kronos-base, NeoQuasar/Kronos-mini, NeoQuasar/Kronos-small, Qwen/Qwen3.5-0.8B, Systran/faster-distil-whisper-large-v3, Systran/faster-whisper-small.en, ggerganov/whisper.cpp, istupakov/parakeet-tdt-0.6b-v3-onnx, sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/clip-ViT-B-32, vikhyatk/moondream2
Warning: You are sending unauthenticated requests to the HF Hub. ...
Loading weights:   0%|          | 0/55 [00:00<?, ?it/s]Loading weights: 100%|##########| 55/55 [00:00<00:00, 6429.57it/s]
downloaded+loaded    26.6 s, dim=384, on disk 134.0 MB
latency              median 8.152 ms per single-sentence encode (96 calls)
{"download_and_load_s":26.6,"downloaded_from_hub":true,"embed_dim":384,"encodes":96,"first_call_ms":18.676,"hub_cache":"~/.cache/huggingface/hub","median_ms_per_sentence":8.1518,"model":"sentence-transformers/paraphrase-MiniLM-L3-v2","ok":true,"on_disk_bytes":140545482,"repos_present_before":14,"torch":"2.11.0+cpu"}
```

Model chosen for this: `sentence-transformers/paraphrase-MiniLM-L3-v2` — Apache-2.0, 61 MB of
weights, and a 3-layer encoder, so it also yields a second latency point (8.15 ms vs 12.48 ms for
L6) for free. It is not a survey candidate and nothing downstream depends on it.

### The workspace is usable for phase 3/4, not just importable

```
{"localalexa": "0.1.0", "rows": 382, "rung5_keys": ["bench", "comparisons", "error_correlation", "fold_assignments", "protocol_notes", "runs"]}
{"python": "3.11.9", "setfit_import": "ok", "setfit_version": "1.2.0", "localalexa_import": "ok",
 "routing_382_type": "list", "routing_382_len": 382,
 "routing_382_first_keys": ["family", "text", "tool"]}
```

`from setfit import SetFitModel, Trainer, TrainingArguments` all resolve under transformers 5.6.2.
That is an import check only — whether SetFit *trains* under transformers 5.x is phase 4's business.

---

## (e) Invocation pattern that works

Local shell → PowerShell 5.1 over ssh. **`$` is interpolated twice**, so the rule that makes this
reliable is: one layer per command, and anything with `$` or nested quotes goes into a scp'd file.

Working patterns, exactly as run:

```bash
# 1. facts / one-layer PowerShell, no $ and no nested quotes
ssh -o ConnectTimeout=8 probe-box 'Get-Command tar.exe | Select-Object -ExpandProperty Source; (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors'

# 2. send a file: to the workspace, or to the home dir
scp -o ConnectTimeout=8 probe.py         probe-box:~/base-model-survey/
scp -o ConnectTimeout=8 03_venv.ps1      probe-box:~/

# 3. run a scp'd script by path (this is the default for anything non-trivial)
ssh -o ConnectTimeout=8 probe-box 'powershell -NoProfile -ExecutionPolicy Bypass -File ~/03_venv.ps1'

# 4. run a python script with the workspace venv  <-- the phase-3/4 workhorse
ssh -o ConnectTimeout=8 probe-box '& ~/base-model-survey/.venv/Scripts/python.exe ~/base-model-survey/probe.py'

# 5. pull a small result back
scp -o ConnectTimeout=8 probe-box:~/base-model-survey/results/foo.json ./results/survey-2026-11-23/
```

Wrap every call in `timeout N` locally if there is any chance of a hang. Things that are **not** to
be done, learned the hard way today:

- `ssh probe-box 'powershell -NoProfile -Command "..."'` with `$` inside — the sshd default shell is
  PowerShell and wraps your command in double quotes, so `$PSVersionTable` is expanded *before*
  PowerShell sees it as a command. It arrives as `System.Collections.Hashtable`.
- `&&` (does not exist in 5.1), `2>nul` (creates a file named `nul`; use `2>$null`, or a scp'd script).
- Any path under `~/mnt/probe-box*`, from either side.
- In a scp'd `.ps1`, a condition continued onto a new line that **starts** with `-or` / `-and` is a
  parse error in 5.1 (`Where-Object : The term '-or' is not recognized`). Put the whole condition on
  one line, or end the line with the operator. This script is also worth sending with CRLF endings
  (`sed -i 's/$/\r/' file.ps1` before `scp`); LF-only files have worked here, but CRLF is native and
  removes one class of doubt from the quoting story.

Convention for later stages: keep python entry points in the workspace root, keep `.ps1` wrappers in
`~/`, and quote every fact back with the script or command that produced it.

---

## (f) What failed, deviated, or needs a decision

1. **The brief's probe could not prove what it was meant to prove.** `all-MiniLM-L6-v2` was already
   cached on this box: the resolved snapshot `1110a243fdf4` has `CreationTime = 2026-09-21T17:02:34`,
   and the process start was 2026-11-23T14:47. Loading it only proved Hub *reachability* (the
   unauthenticated-request warning is evidence of an HTTP call, and nothing more). **Not fixed by
   assumption** — `07_download_proof.py` downloads a model that is provably absent from the cache.
   The download leg now has its own evidence: 26.6 s, 134.0 MB on disk, dim 384.

2. **`onnxruntime` was already there, so the "skip it if it does not install cleanly" branch never
   ran.** Version 1.20.1 is inherited from the system interpreter. Nothing was installed, nothing
   failed. If a later stage needs a specific onnxruntime build, that is a new decision, not a
   leftover from this one.

3. **Deviation: the venv's pip is 24.0, the system's is 25.3.** `--system-site-packages` does not
   inherit `pip`, so `ensurepip` supplied a fresh 24.0 inside the venv. Everything installed fine
   with it. Left alone deliberately — the brief said not to touch the interpreter's stack, and a pip
   upgrade is the kind of change that silently re-resolves other packages.

4. **Deviation: `pyproject.toml` declares `requires-python = ">=3.12"`; the box is 3.11.9.** So
   `pip install -e .` cannot work here, and it was not attempted. The workspace imports the package
   the plain way instead (`sys.path.insert(0, root/'src')`), verified: `localalexa 0.1.0` imports and
   reads `routing_382.json` (382 rows, keys `family/text/tool`) and `rung5-ensemble.json`. **Worth a
   decision before phase 4**: either the survey accepts `PYTHONPATH=src` on both machines, or the
   project's floor is reconciled with the measurement box.

5. **`setfit 1.2.0` under `transformers 5.6.2` imports cleanly**, including `Trainer` and
   `TrainingArguments`. Untested: whether SetFit actually *trains* under transformers 5. This is a
   real compatibility risk for phase 4 and the cheap time to find out is before an overnight run, not
   during one. Flagged, not investigated — it is out of scope for an environment smoke test.

6. **Pre-existing dependency conflicts in the system interpreter, untouched.**
   `pip install` printed a long block of them; the ones that matter to this survey: none directly,
   but note `torchaudio 2.9.1+cpu requires torch==2.9.1, but you have torch 2.11.0`. This box's
   system site has a large unrelated install (`open-webui` wants `sentence-transformers==4.1.0`, and
   gets 5.4.1; `unstructured` wants `numpy<2`, and gets 2.4.4). Those warnings come from inheriting
   that site, not from anything installed today, and none of it is used by the survey. **Do not
   "fix" these** — pinning against them would break the machine's primary installs.

7. **The hub cache directory is 1019 MB for an 87 MB model.** Two snapshots: `1110a243fdf4`
   (87.3 MB, plain PyTorch) and `c9745ed1d9f2` (931.7 MB, from 2025-09-08, containing `onnx/` and
   `openvino/`) left by earlier work on this box. `hf_hub` resolved to the plain one. Nothing was
   deleted — it is not this task's disk to reclaim, and 211 GB is free. `probe.py` therefore reports
   both `model_size_bytes` (the snapshot actually loaded) and the whole-directory figure, so a later
   reader cannot mistake one for the other.

8. **`src/localalexa.egg-info` arrived with the tar.** It is an untracked, gitignored build artifact
   from a local venv; the tar named only directories, so it came along. Removed remotely — a stale
   egg-info is exactly what makes pip resolve against old metadata.

9. **Three failed attempts before the pattern was clear.** Two were
   `ssh probe-box 'powershell -NoProfile -Command "...$PSVersionTable..."'` and returned
   `ParserError: An expression was expected after '('` with the variable already substituted to
   `System.Collections.Hashtable`. The third was a scp'd `.ps1` whose `Where-Object` condition put
   `-or` at the start of a continuation line: `Where-Object : The term '-or' is not recognized as the
   name of a cmdlet`. Cause and fix for all three are in (e); recorded because the next person to hit
   this box will write the same one-liner.

10. **The remote tree is a snapshot of 14:42, and the repo moved after that.** A concurrent session
    was working in the same repo while this ran: it committed `d581e5a` at 14:45 (the files that were
    untracked when this session began) and created `experiments/probe-box/` at 14:46-14:50 — the two
    overnight jobs from `docs/probe-box-overnight-brief-2026-11-23.md`. The tar was built at 14:42, so
    **the remote workspace does not contain `experiments/probe-box/`**, and it holds the branch as of
    `e4c1d74`-era state rather than `d581e5a`. Nothing was re-synced and nothing was touched: that
    work is not this stage's to move. Whoever runs phase 3/4 should re-sync before depending on the
    remote tree, or run those jobs on their own workspace.

11. **No credentials were used or recorded.** The box has no `HF_TOKEN` set — that is what the
    "unauthenticated requests" warning means. Anonymous pulls are rate-limited, which matters only if
    a later stage downloads many models at once; if that becomes a problem it is an operator decision,
    not something to work around by embedding a token in a script.

---

### What this stage does not establish

- No accuracy number, for any model. Nothing here measured a classifier.
- No latency comparison against the ladder. The probe's 12.5 ms is a batch-1 smoke test and the
  ladder's 7.75 ms is a different protocol; they are not the same quantity and the gap should not be
  quoted as a regression.
- No SetFit *training* run under transformers 5. (f.5).
- Nothing on the iGPU. CPU only, as instructed.
