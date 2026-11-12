$ErrorActionPreference = 'Stop'
$root = Join-Path ([IO.Path]::GetTempPath()) ("climate overnight ' " + [guid]::NewGuid())
$runner = Join-Path $PSScriptRoot 'run_climate_overnight.ps1'
$python = "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe"
New-Item -ItemType Directory -Path $root | Out-Null

function Assert-True($Condition, $Message) {
    if (-not $Condition) { throw $Message }
}

function Invoke-Runner([string[]]$Extra) {
    $ErrorActionPreference = 'Continue'
    & "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File $runner `
        -Python $python -SourceDir "$root\source" -OutputDir "$root\result" @Extra 2>&1 |
        ForEach-Object { "$_" }
    return $LASTEXITCODE
}

try {
    New-Item -ItemType Directory -Path "$root\source", "$root\result" | Out-Null
    $records = @(foreach ($year in 1996..2025) {
        $path = "$root\source\$year.nc"
        Set-Content -LiteralPath $path -Value 'fixture'
        @{ year = $year; netcdf = $path; bytes = (Get-Item -LiteralPath $path).Length }
    })
    @{ status = 'complete'; archives = $records } | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath "$root\source\manifest.json"
    @{ status = 'complete'; period = '1996-2025' } | ConvertTo-Json |
        Set-Content -LiteralPath "$root\result\analysis.json"
    New-Item -ItemType Directory -Path "$root\result\k8", "$root\result\k32" | Out-Null
    foreach ($name in @(
        'k8\climate_regions_1km.tif', 'k32\climate_regions_1km.tif',
        'k8\analysis.json', 'k32\analysis.json',
        'summer_winter_temperature_amplitude_c_1km.tif', 'frost_days_per_year_1km.tif'
    )) { Set-Content -LiteralPath "$root\result\$name" -Value 'fixture' }
    $output = @(Invoke-Runner @('-Foreground'))
    Assert-True ($output[-1] -eq 0) 'Completed run did not succeed.'
    $status = Get-Content -LiteralPath "$root\result\overnight_status.json" -Raw | ConvertFrom-Json
    Assert-True ($status.state -eq 'complete') 'Completed run did not write success status.'

    $output = @(Invoke-Runner @())
    Assert-True ($output[-1] -eq 0) 'Background launch failed.'
    $deadline = (Get-Date).AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 250
        $logs = @(Get-ChildItem -LiteralPath "$root\result" -Filter 'overnight_*.log' |
            Where-Object Name -NotLike '*.errors.log')
        $finished = $logs.Count -gt 0 -and
            ((Get-Content -LiteralPath $logs[0].FullName -Raw) -match 'Climate outputs are ready')
    } until ($finished -or (Get-Date) -gt $deadline)
    Assert-True $finished 'Independent worker did not finish in 30 seconds.'
    # The fixture directory contains both spaces and an apostrophe.
    Assert-True ($logs.Count -eq 1) 'Expected a separate timestamped progress log.'

    $handle = [IO.File]::Open("$root\result\overnight.lock", 'OpenOrCreate', 'ReadWrite', 'None')
    try {
        $output = @(Invoke-Runner @('-Foreground') 2>&1)
        Assert-True ($output[-1] -eq 1) 'Concurrent worker was not rejected.'
    } finally { $handle.Dispose() }

    Remove-Item -LiteralPath "$root\result\frost_days_per_year_1km.tif"
    $output = @(Invoke-Runner @('-Foreground') 2>&1)
    Assert-True ($output[-1] -eq 1) 'Missing required output did not fail.'
    $status = Get-Content -LiteralPath "$root\result\overnight_status.json" -Raw | ConvertFrom-Json
    Assert-True ($status.state -eq 'failed') 'Failure was not recorded.'

    $stub = "$root\fail.cmd"
    Set-Content -LiteralPath $stub -Value "@echo off`r`necho simulated Python failure`r`nexit /b 7"
    $ErrorActionPreference = 'Continue'
    & "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File $runner -Foreground `
        -Python $stub -SourceDir "$root\missing-source" -OutputDir "$root\failed-stage" 2>&1 | Out-Null
    $ErrorActionPreference = 'Stop'
    Assert-True ($LASTEXITCODE -eq 1) 'Native Python failure was not propagated.'
    $status = Get-Content -LiteralPath "$root\failed-stage\overnight_status.json" -Raw | ConvertFrom-Json
    Assert-True ($status.state -eq 'failed' -and $status.message -match 'code 7') `
        'Native exit code was not recorded in status.'
    Write-Output 'PASS: foreground, background, quoted paths, duplicate lock, missing output, native failure.'
} finally {
    Remove-Item -LiteralPath $root -Recurse -Force
}
