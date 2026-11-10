$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $testRoot = Join-Path $projectRoot ('data/test-' + [guid]::NewGuid().ToString('N'))
    & .venv/Scripts/python.exe -m pytest -q --basetemp $testRoot
    exit $LASTEXITCODE
} finally { Pop-Location }
