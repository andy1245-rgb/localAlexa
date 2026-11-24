# 02_extract.ps1 - unpack the repo subset into the workspace and verify it.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File ~/base-model-survey/02_extract.ps1
$ErrorActionPreference = 'Stop'
$root = '~/base-model-survey'
$tgz  = '~/localalexa-subset.tgz'

Write-Output ('archive         = ' + $tgz + ' bytes=' + (Get-Item $tgz).Length)

& tar.exe -xzf $tgz -C $root
if ($LASTEXITCODE -ne 0) { throw ('tar extract failed, exit ' + $LASTEXITCODE) }
Write-Output 'extract         = ok'

$files = Get-ChildItem -Path $root -Recurse -File
Write-Output ('remote_files    = ' + $files.Count)

$must = @(
    'data\routing_382.json',
    'results\ladder\rung5-ensemble.json',
    'pyproject.toml'
)
foreach ($rel in $must) {
    $p = Join-Path $root $rel
    if (Test-Path $p) {
        Write-Output ('present ' + $rel + ' = ' + (Get-Item $p).Length + ' bytes')
    } else {
        Write-Output ('MISSING ' + $rel)
    }
}

# Anything that should not have travelled.
foreach ($bad in @('.git', 'checkpoints', '.venv', '__pycache__')) {
    $p = Join-Path $root $bad
    $hits = @(Get-ChildItem -Path $root -Recurse -Force -Filter $bad -Directory -ErrorAction SilentlyContinue)
    Write-Output ('unexpected ' + $bad + ' = ' + $hits.Count)
}

$tot = (Get-ChildItem -Path $root -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Output ('remote_total_mb = ' + [math]::Round($tot / 1MB, 2))
