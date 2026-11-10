param([switch]$DownloadModels, [switch]$RegisterCodex)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv before running setup.' }
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        uv --system-certs python install 3.12
        if ($LASTEXITCODE) { throw 'Python installation failed.' }
        $managedPython = & uv --system-certs python find --managed-python --system --no-project 3.12
        if ($LASTEXITCODE -or -not (Test-Path -LiteralPath $managedPython)) { throw 'Cannot find the managed Python 3.12 installation.' }
        uv venv --python $managedPython .venv
        if ($LASTEXITCODE) { throw 'Virtual environment creation failed.' }
    }
    uv --system-certs pip install --python .venv/Scripts/python.exe 'torch==2.14.1' 'torchvision==0.29.1' --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed.' }
    uv --system-certs pip install --python .venv/Scripts/python.exe -r requirements.lock.txt
    if ($LASTEXITCODE) { throw 'Locked dependency installation failed.' }
    uv --system-certs pip install --python .venv/Scripts/python.exe -e '.[inference,test]'
    if ($LASTEXITCODE) { throw 'Dependency installation failed.' }
    & .venv/Scripts/python.exe -m flux_image install-cli
    if ($LASTEXITCODE) { throw 'CLI installation failed.' }
    if ($RegisterCodex) {
        & .venv/Scripts/python.exe -m flux_image register-codex
        if ($LASTEXITCODE) { throw 'Codex MCP registration failed.' }
    }
    if ($DownloadModels) {
        & .venv/Scripts/python.exe -m flux_image setup-models
        if ($LASTEXITCODE) { throw 'Model download or verification failed; resolve access/network errors and rerun setup-models.' }
    }
} finally { Pop-Location }
