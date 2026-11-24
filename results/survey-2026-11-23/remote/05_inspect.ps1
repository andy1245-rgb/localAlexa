# 05_inspect.ps1 - explain the HF cache size, and confirm setfit + the repo import on this box.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File ~/base-model-survey/05_inspect.ps1
$ErrorActionPreference = 'Stop'
$root   = '~/base-model-survey'
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
$repo   = '~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'

Write-Output '--- HF repo cache: snapshots present ---'
$snaps = Get-ChildItem -Path (Join-Path $repo 'snapshots') -Directory -ErrorAction SilentlyContinue
foreach ($s in $snaps) {
    $bytes = (Get-ChildItem -Path $s.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum
    $dirs  = @(Get-ChildItem -Path $s.FullName -Directory | Select-Object -ExpandProperty Name) -join ','
    Write-Output ('  ' + $s.Name.Substring(0, 12) + '  ' +
        [math]::Round($bytes / 1MB, 1) + ' MB  subdirs=[' + $dirs + ']')
}
$blobs = (Get-ChildItem -Path (Join-Path $repo 'blobs') -File -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum
Write-Output ('  blobs/ total ' + [math]::Round($blobs / 1MB, 1) + ' MB (deduplicated store)')

Write-Output ''
Write-Output '--- import checks (venv) ---'
$check = @'
import json, sys
from pathlib import Path
root = Path(r"~/base-model-survey")
sys.path.insert(0, str(root / "src"))

out = {"python": sys.version.split()[0]}
try:
    import setfit
    from setfit import SetFitModel, Trainer, TrainingArguments
    out["setfit_import"] = "ok"
    out["setfit_version"] = getattr(setfit, "__version__", "?")
except Exception as e:
    out["setfit_import"] = type(e).__name__ + ": " + str(e)[:300]

try:
    import localalexa
    out["localalexa_import"] = "ok"
except Exception as e:
    out["localalexa_import"] = type(e).__name__ + ": " + str(e)[:300]

try:
    rows = json.loads((root / "data" / "routing_382.json").read_text(encoding="utf-8"))
    out["routing_382_type"] = type(rows).__name__
    out["routing_382_len"] = len(rows)
    if isinstance(rows, list) and rows:
        out["routing_382_first_keys"] = sorted(rows[0].keys())[:8]
except Exception as e:
    out["routing_382"] = type(e).__name__ + ": " + str(e)[:300]

print(json.dumps(out, indent=2))
'@
$f = Join-Path $root '_import_check.py'
Set-Content -Path $f -Value $check -Encoding ASCII
& $venvPy $f
$code = $LASTEXITCODE
Remove-Item $f -Force
Write-Output ('import_check_exit = ' + $code)
