# 08_finalize.ps1 - remove stale build metadata, archive the setup scripts, print the final state.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File ~/08_finalize.ps1
$ErrorActionPreference = 'Stop'
$root  = '~/base-model-survey'
$setup = Join-Path $root 'setup'
$venvPy = Join-Path $root '.venv\Scripts\python.exe'

# Stale local egg-info came in with the tar; pip would resolve against its old metadata.
$egg = Join-Path $root 'src\localalexa.egg-info'
if (Test-Path $egg) {
    Remove-Item -Path $egg -Recurse -Force
    Write-Output 'removed             src\localalexa.egg-info (untracked build artifact swept in by tar)'
}

New-Item -ItemType Directory -Force -Path $setup | Out-Null
Get-ChildItem -Path '~' -File -Filter '0*.ps1' | ForEach-Object {
    Copy-Item $_.FullName -Destination $setup -Force
    Write-Output ('archived            ' + $_.Name)
}

Write-Output ''
Write-Output '--- workspace ---'
Write-Output ('root                ' + $root)
Write-Output ('venv_python         ' + $venvPy)
Get-ChildItem -Path $root -Directory | ForEach-Object {
    Write-Output ('  dir  ' + $_.Name)
}
Get-ChildItem -Path $root -File | ForEach-Object {
    Write-Output ('  file ' + $_.Name + '  ' + $_.Length)
}

Write-Output ''
Write-Output '--- reproducibility check: repo imports from the workspace, no install ---'
$chk = @'
import sys, json
from pathlib import Path
root = Path(r"~/base-model-survey")
sys.path.insert(0, str(root / "src"))
import localalexa
rows = json.loads((root / "data" / "routing_382.json").read_text(encoding="utf-8"))
lad = json.loads((root / "results" / "ladder" / "rung5-ensemble.json").read_text(encoding="utf-8"))
print(json.dumps({"localalexa": getattr(localalexa, "__version__", "ok"),
                  "rows": len(rows),
                  "rung5_keys": sorted(lad.keys())[:6] if isinstance(lad, dict) else "n/a"}))
'@
$f = Join-Path $root '_repro.py'
Set-Content -Path $f -Value $chk -Encoding ASCII
& $venvPy $f
$code = $LASTEXITCODE
Remove-Item $f -Force
Write-Output ('repro_exit          ' + $code)

$n = (Get-ChildItem -Path $root -Recurse -File | Measure-Object).Count
$mb = [math]::Round((Get-ChildItem -Path $root -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Output ('workspace_files     ' + $n + '  (' + $mb + ' MB, venv included)')
