# 03_venv.ps1 - create the venv over system site-packages and report what is already visible.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File ~/base-model-survey/03_venv.ps1
$ErrorActionPreference = 'Stop'
$root   = '~/base-model-survey'
$sysPy  = '~/AppData/Local/Programs/Python/Python311/python.exe'
$venv   = Join-Path $root '.venv'
$venvPy = Join-Path $venv 'Scripts\python.exe'

Write-Output ('sys_python      = ' + $sysPy)

if (Test-Path $venvPy) {
    Write-Output 'venv            = already exists, reusing'
} else {
    & $sysPy -m venv --system-site-packages $venv
    if ($LASTEXITCODE -ne 0) { throw ('venv creation failed, exit ' + $LASTEXITCODE) }
    Write-Output 'venv            = created (--system-site-packages)'
}

Write-Output ('venv_python     = ' + $venvPy)
Write-Output ('venv_version    = ' + (& $venvPy -c "import sys;print(sys.version.replace(chr(10),' '))"))
Write-Output ('pip             = ' + (& $venvPy -m pip --version))

$probe = @'
import importlib.metadata as md
import json
names = ["torch", "transformers", "sentence-transformers", "setfit", "scikit-learn",
         "pandas", "numpy", "onnxruntime", "huggingface-hub", "tokenizers", "scipy"]
out = {}
for n in names:
    try:
        out[n] = md.version(n)
    except Exception:
        out[n] = None
print(json.dumps(out, indent=0))
'@
$probeFile = Join-Path $root '_probe_versions.py'
Set-Content -Path $probeFile -Value $probe -Encoding ASCII
Write-Output '--- installed metadata as seen by the venv ---'
& $venvPy $probeFile
if ($LASTEXITCODE -ne 0) { throw ('version probe failed, exit ' + $LASTEXITCODE) }
Remove-Item $probeFile -Force
