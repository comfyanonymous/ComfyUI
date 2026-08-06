$ErrorActionPreference = 'Stop'

$comfyRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$customNodes = Join-Path $comfyRoot 'custom_nodes'

$repositories = @(
    @{
        Name = 'ComfyUI-MiniMax-H3-Turbo'
        Url = 'https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git'
        Commit = '96cc1ddc001617da132dd73f31cd43666bf1d8d4'
    },
    @{
        Name = 'ComfyUI-sol-attn'
        Url = 'https://github.com/Saganaki22/ComfyUI-sol-attn.git'
        Commit = 'e2fc225f8642585cfa11a31d52fe7b2db7290efa'
    },
    @{
        Name = 'ComfyUI-Spectrum-MiniMax-H3'
        Url = 'https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git'
        Commit = '85ec1da66277e893079ecd46e32cc865c56cfe53'
    }
)

foreach ($repository in $repositories) {
    $destination = Join-Path $customNodes $repository.Name
    if (-not (Test-Path -LiteralPath (Join-Path $destination '.git'))) {
        if (Test-Path -LiteralPath $destination) {
            throw "Existing non-Git directory blocks installation: $destination"
        }
        Write-Host "Cloning $($repository.Name)..." -ForegroundColor Cyan
        & git clone --filter=blob:none --no-checkout $repository.Url $destination
        if ($LASTEXITCODE -ne 0) {
            throw "Clone failed for $($repository.Name)"
        }
    }

    Write-Host "Pinning $($repository.Name) to $($repository.Commit)..."
    & git -C $destination fetch --depth 1 origin $repository.Commit
    if ($LASTEXITCODE -ne 0) {
        throw "Fetch failed for $($repository.Name)"
    }
    & git -C $destination checkout --detach $repository.Commit
    if ($LASTEXITCODE -ne 0) {
        throw "Checkout failed for $($repository.Name)"
    }
}

Write-Host 'RTX 3060 MiniMax-H3 custom nodes are installed and pinned.' -ForegroundColor Green
