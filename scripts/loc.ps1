# Write reports/loc.md with cloc.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$cloc = Join-Path $env:LOCALAPPDATA "deeplearn-tools\cloc.exe"
if (-not (Test-Path $cloc))
{
    $found = Get-Command cloc -ErrorAction SilentlyContinue
    if ($found)
    {
        $cloc = $found.Source
    }
    else
    {
        throw "cloc was not found. Install it, or place cloc.exe at $env:LOCALAPPDATA\deeplearn-tools\cloc.exe."
    }
}

python (Join-Path $PSScriptRoot "loc_report.py") --cloc $cloc
if ($LASTEXITCODE -ne 0)
{
    exit $LASTEXITCODE
}
