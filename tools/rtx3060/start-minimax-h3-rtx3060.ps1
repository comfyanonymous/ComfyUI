param(
    [switch]$IncludeSpectrum
)

$ErrorActionPreference = 'Stop'

$comfyRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $comfyRoot 'venv/Scripts/python.exe'
$main = Join-Path $comfyRoot 'main.py'
$optimizer = Join-Path $PSScriptRoot 'apply-transformers-startup-cache.ps1'

if (-not (Test-Path -LiteralPath $python)) {
    throw "ComfyUI Python not found: $python"
}

$listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Host 'ComfyUI is already listening at http://127.0.0.1:8188' -ForegroundColor Yellow
    exit 0
}

& $optimizer

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:CUDA_MODULE_LOADING = 'LAZY'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

$whitelist = @(
    'ComfyUI-MiniMax-H3-Turbo',
    'ComfyUI-sol-attn',
    'ComfyMath'
)
if ($IncludeSpectrum) {
    $whitelist += 'ComfyUI-Spectrum-MiniMax-H3'
}

$arguments = @(
    '-X', 'utf8',
    '-u', $main,
    '--listen', '127.0.0.1',
    '--port', '8188',
    '--preview-method', 'none',
    '--use-pytorch-cross-attention',
    '--cache-none',
    '--reserve-vram', '1.0',
    '--disable-pinned-memory',
    '--async-offload', '1',
    '--disable-all-custom-nodes',
    '--whitelist-custom-nodes'
) + $whitelist

Write-Host 'Starting MiniMax-H3 RTX 3060 mode...' -ForegroundColor Cyan
Write-Host ('Custom-node whitelist: ' + ($whitelist -join ', '))
Set-Location -LiteralPath $comfyRoot
& $python @arguments
exit $LASTEXITCODE
