# 09_counts.ps1 - exact file counts per group, and clear generated __pycache__.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File ~/09_counts.ps1
$ErrorActionPreference = 'Stop'
$root = '~/base-model-survey'

$pyc = @(Get-ChildItem -Path $root -Recurse -Directory -Filter '__pycache__' | Where-Object { $_.FullName -notmatch '\\\.venv\\' })
foreach ($d in $pyc) { Remove-Item -Path $d.FullName -Recurse -Force }
Write-Output ('removed __pycache__ dirs = ' + $pyc.Count)

$all = @(Get-ChildItem -Path $root -Recurse -File | Where-Object { $_.FullName -notmatch '\\\.venv\\' })

$subset = 0; $rootPy = 0; $setup = 0; $other = 0
$rootPyNames = @(); $otherNames = @()
$subsetBytes = 0
foreach ($f in $all) {
    $rel = $f.FullName.Substring($root.Length + 1)
    $isSubset = ($rel -match '^(data|src|experiments|tests|docs)\\') -or ($rel -match '^results\\ladder\\') -or ($rel -eq 'pyproject.toml')
    $isRootPy = $rel -match '^[^\\]+\.py$'
    $isSetup  = $rel -match '^setup\\'
    if ($isSubset) { $subset++; $subsetBytes += $f.Length }
    elseif ($isRootPy) { $rootPy++; $rootPyNames += $f.Name }
    elseif ($isSetup) { $setup++ }
    else { $other++; $otherNames += $f.Name }
}

Write-Output ('repo_subset_files   = ' + $subset + '  (' + [math]::Round($subsetBytes / 1MB, 2) + ' MB)')
Write-Output ('root_py_files       = ' + $rootPy + '  [' + ($rootPyNames -join ' ') + ']')
Write-Output ('setup_ps1_files     = ' + $setup)
Write-Output ('other_files         = ' + $other + '  [' + ($otherNames -join ' ') + ']')
Write-Output ('total_excl_venv     = ' + $all.Count)
Write-Output ('total_excl_venv_mb  = ' + [math]::Round(($all | Measure-Object -Property Length -Sum).Sum / 1MB, 2))
Write-Output ('venv_files          = ' + @(Get-ChildItem -Path (Join-Path $root '.venv') -Recurse -File).Count)

foreach ($must in @('data\routing_382.json', 'results\ladder\rung5-ensemble.json', 'pyproject.toml')) {
    $p = Join-Path $root $must
    $ok = Test-Path $p
    $len = 0
    if ($ok) { $len = (Get-Item $p).Length }
    Write-Output ('check ' + $must + ' present=' + $ok + ' bytes=' + $len)
}
