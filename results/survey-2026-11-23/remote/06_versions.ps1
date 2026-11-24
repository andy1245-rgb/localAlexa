# 06_versions.ps1 - exact version strings + whether the Hub snapshot was fetched during this run.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\andre\base-model-survey\06_versions.ps1
$ErrorActionPreference = 'Stop'
$root   = 'C:\Users\andre\base-model-survey'
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
$sysPy  = 'C:\Users\andre\AppData\Local\Programs\Python\Python311\python.exe'
$repo   = 'C:\Users\andre\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2'

Write-Output ('now                 = ' + (Get-Date -Format o))
Write-Output ('venv_pip            = ' + (& $venvPy -m pip --version))
Write-Output ('sys_pip             = ' + (& $sysPy -m pip --version))

$v = @'
import json
import torch, transformers, sentence_transformers, setfit, sklearn
print(json.dumps({
    "torch": torch.__version__,
    "torch_cpu_only_build": torch.version.cuda is None,
    "transformers": transformers.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "setfit": setfit.__version__,
    "scikit_learn": sklearn.__version__,
}, indent=2))
'@
$f = Join-Path $root '_v.py'
Set-Content -Path $f -Value $v -Encoding ASCII
& $venvPy $f
Remove-Item $f -Force

Write-Output '--- snapshot timestamps (did the Hub deliver today?) ---'
foreach ($s in (Get-ChildItem -Path (Join-Path $repo 'snapshots') -Directory)) {
    $created = (Get-Item $s.FullName).CreationTime.ToString('o')
    $model = Join-Path $s.FullName 'model.safetensors'
    $mt = if (Test-Path $model) { (Get-Item $model).LastWriteTime.ToString('o') } else { 'no model.safetensors' }
    Write-Output ('  ' + $s.Name.Substring(0, 12) + '  created=' + $created + '  model.safetensors=' + $mt)
}
Write-Output '--- files in the resolved snapshot ---'
$snap = Join-Path $repo 'snapshots\1110a243fdf4a1d252b0f33f5c4c852e1c59c4c5'
if (Test-Path $snap) {
    Get-ChildItem -Path $snap -Recurse -File | ForEach-Object {
        Write-Output ('  ' + $_.Name + ' ' + $_.Length)
    }
}
