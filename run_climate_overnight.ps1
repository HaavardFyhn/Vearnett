[CmdletBinding()]
param(
    [string]$Python = "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe",
    [string]$SourceDir,
    [string]$OutputDir,
    [ValidateRange(1, 64)][int]$Threads = 4,
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'
if (-not $SourceDir) { $SourceDir = Join-Path $PSScriptRoot 'outputs\senorge_norway_1996_2025' }
if (-not $OutputDir) { $OutputDir = Join-Path $PSScriptRoot 'outputs\climate_regions_1996_2025' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$SourceDir = [IO.Path]::GetFullPath($SourceDir)
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

if (-not $Foreground) {
    # EncodedCommand preserves spaces, apostrophes and non-ASCII Windows paths.
    $quote = { param($value) "'" + $value.Replace("'", "''") + "'" }
    $command = "& $( & $quote $PSCommandPath ) -Foreground -Python $( & $quote $Python )" +
        " -SourceDir $( & $quote $SourceDir ) -OutputDir $( & $quote $OutputDir ) -Threads $Threads"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stdout = Join-Path $OutputDir "overnight_$stamp.log"
    $stderr = Join-Path $OutputDir "overnight_$stamp.errors.log"
    $process = Start-Process -FilePath "$PSHOME\powershell.exe" -ArgumentList @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encoded
    ) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    Write-Output "Background worker PID: $($process.Id)"
    Write-Output "Progress log: $stdout"
    Write-Output "Error log: $stderr"
    Write-Output "Status: $(Join-Path $OutputDir 'overnight_status.json')"
    Write-Output 'You can close the terminal and VS Code. Keep Windows awake and plugged in.'
    exit 0
}

$lock = $null
$statusPath = Join-Path $OutputDir 'overnight_status.json'
$started = (Get-Date).ToString('o')
$stage = 'starting'

function Write-RunStatus([string]$State, [string]$Message) {
    $pending = "$statusPath.pending"
    @{
        state = $State; stage = $stage; message = $Message
        pid = $PID; started = $started; updated = (Get-Date).ToString('o')
        source = $SourceDir; output = $OutputDir
    } | ConvertTo-Json | Set-Content -LiteralPath $pending -Encoding UTF8
    Move-Item -LiteralPath $pending -Destination $statusPath -Force
    Write-Output "[$((Get-Date).ToString('s'))] $State / ${stage}: $Message"
}

function Invoke-PythonStep([string[]]$Arguments) {
    # Merge native stderr for readable, chronological progress without PS5 error wrapping.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Python @Arguments 2>&1 | ForEach-Object { Write-Output "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "Python stage '$stage' exited with code $code. See the progress log."
    }
}

try {
    try {
        $lock = [IO.File]::Open((Join-Path $OutputDir 'overnight.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch [IO.IOException] {
        throw "Cannot acquire output lock; another overnight worker may be using $OutputDir. $($_.Exception.Message)"
    }
    Set-Location -LiteralPath $PSScriptRoot
    $env:PYTHONUNBUFFERED = '1'
    $env:LOKY_MAX_CPU_COUNT = "$Threads"
    $stage = 'download'
    Write-RunStatus 'running' 'Checking retained annual archives.'
    $manifestPath = Join-Path $SourceDir 'manifest.json'
    $ready = $false
    if (Test-Path -LiteralPath $manifestPath) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $years = @($manifest.archives | ForEach-Object { $_.year })
        $ready = $manifest.status -eq 'complete' -and ($years -join ',') -eq ((1996..2025) -join ',')
        if ($ready) {
            foreach ($archive in $manifest.archives) {
                if (-not (Test-Path -LiteralPath $archive.netcdf -PathType Leaf)) {
                    $ready = $false
                    break
                }
                if ((Get-Item -LiteralPath $archive.netcdf).Length -ne $archive.bytes) {
                    throw "Archive size mismatch: $($archive.netcdf). Restore or inspect the original before retrying."
                }
            }
        }
    }
    if (-not $ready) {
        Invoke-PythonStep @(
            '-u', (Join-Path $PSScriptRoot 'download_senorge_norway.py'),
            '--annual', '--start', '1996-01-01', '--end', '2025-12-31',
            '--output-dir', $SourceDir, '--reuse-dir',
            (Join-Path $PSScriptRoot 'outputs\senorge_norway_2021_2025\netcdf')
        )
    } else {
        Write-Output 'All 30 previously verified originals exist with matching sizes; skipping download.'
    }
    $stage = 'analysis'
    Write-RunStatus 'running' 'Running climate regions, or checking already completed results.'
    $analysisPath = Join-Path $OutputDir 'analysis.json'
    if (Test-Path -LiteralPath $analysisPath) {
        $analysis = Get-Content -LiteralPath $analysisPath -Raw | ConvertFrom-Json
        if ($analysis.status -ne 'complete' -or $analysis.period -ne '1996-2025') {
            throw 'Unexpected analysis receipt. Inspect it or choose a new output directory.'
        }
        Write-Output 'Analysis already complete; not rerunning or overwriting it.'
    } else {
        Invoke-PythonStep @(
            '-u', (Join-Path $PSScriptRoot 'climate_regions.py'), '--source', $SourceDir,
            '--output-dir', $OutputDir, '--threads', "$Threads"
        )
    }
    $stage = 'checking-outputs'
    foreach ($relative in @(
        'analysis.json', 'k8\climate_regions_1km.tif', 'k32\climate_regions_1km.tif',
        'k8\analysis.json', 'k32\analysis.json',
        'summer_winter_temperature_amplitude_c_1km.tif', 'frost_days_per_year_1km.tif'
    )) {
        $path = Join-Path $OutputDir $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
            throw "Required output missing or empty: $path"
        }
    }
    Write-RunStatus 'complete' 'Climate outputs are ready. Originals retained.'
} catch {
    if ($null -ne $lock) {
        Write-RunStatus 'failed' $_.Exception.Message
    }
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
} finally {
    if ($null -ne $lock) { $lock.Dispose() }
}
