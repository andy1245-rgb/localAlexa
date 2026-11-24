# 04_pip.ps1 - install only what the venv cannot already see, then record exact resolved versions.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\andre\base-model-survey\04_pip.ps1
$ErrorActionPreference = 'Stop'
$root   = 'C:\Users\andre\base-model-survey'
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
$log    = Join-Path $root 'pip-install.log'

# sentence-transformers / scikit-learn / pandas / onnxruntime / torch / transformers are all
# inherited from the native interpreter via --system-site-packages. setfit is the only gap.
$want = @('setfit')

Start-Transcript -Path $log -Force | Out-Null
Write-Output ('venv_python     = ' + $venvPy)
Write-Output ('pip             = ' + (& $venvPy -m pip --version))
foreach ($pkg in $want) {
    Write-Output ('--- pip install ' + $pkg + ' ---')
    & $venvPy -m pip install --disable-pip-version-check --no-input $pkg
    Write-Output ('pip_exit_' + $pkg + '   = ' + $LASTEXITCODE)
}
Stop-Transcript | Out-Null

Write-Output '--- freeze (venv only, no inherited dists) ---'
& $venvPy -m pip freeze --local
Write-Output ('freeze_exit     = ' + $LASTEXITCODE)
