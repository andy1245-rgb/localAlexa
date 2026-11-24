# 01_prep.ps1 - create the remote workspace and print box facts.
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\andre\base-model-survey\01_prep.ps1
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\andre\base-model-survey'

Write-Output ('date            = ' + (Get-Date -Format o))
Write-Output ('user            = ' + $env:USERNAME)
Write-Output ('pwd             = ' + (Get-Location).Path)
Write-Output ('target          = ' + $root)

if (Test-Path $root) {
    $n = (Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Output ('target_exists   = yes (files=' + $n + ')')
} else {
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    Write-Output 'target_exists   = created'
}

# Sibling dirs must not be touched; report only.
foreach ($sib in @('alexa-finetune', 'base-model-survey')) {
    $p = 'C:\Users\andre\' + $sib
    Write-Output ('sibling ' + $sib + ' = ' + (Test-Path $p))
}

Write-Output ('powershell      = ' + $PSVersionTable.PSVersion.ToString())
Write-Output ('tar             = ' + (Get-Command tar.exe -ErrorAction SilentlyContinue).Source)
Write-Output ('logical_cpus    = ' + (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors)
$mem = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
Write-Output ('ram_gb          = ' + [math]::Round($mem / 1GB, 2))
$cpu = (Get-CimInstance Win32_Processor).Name
Write-Output ('cpu             = ' + $cpu)
foreach ($d in (Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -match '^[CD]$' })) {
    Write-Output ('drive ' + $d.Name + ' free_gb = ' + [math]::Round($d.Free / 1GB, 2))
}

$py = 'C:\Users\andre\AppData\Local\Programs\Python\Python311\python.exe'
Write-Output ('python_exe      = ' + $py + ' exists=' + (Test-Path $py))
