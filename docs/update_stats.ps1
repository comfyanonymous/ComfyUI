# ComfyUI Documentation Stats Updater
# Fetches fresh statistics from Civitai API and updates documentation files
# Usage: .\update_stats.ps1 [-Category "POSES"] [-Verbose] [-DryRun]

param(
    [string]$Category = "",      # Optional: update only specific category folder
    [switch]$DryRun,             # Show what would be updated without making changes
    [switch]$Verbose,            # Show detailed progress
    [int]$DelayMs = 500          # Delay between API calls (ms) to avoid rate limiting
)

$docsPath = "C:\Users\spoko\www\ai\ComfyUI\docs"
$thumbs = [char]::ConvertFromUtf32(0x1F44D)
$star1 = [char]::ConvertFromUtf32(0x2B50)
$stats = @{
    FilesProcessed = 0
    FilesUpdated = 0
    FilesSkipped = 0
    ApiErrors = 0
}

function Get-CivitaiModelId {
    param([string]$Content)

    # Match Civitai URL pattern: https://civitai.com/models/XXXXXXX/...
    if ($Content -match 'https://civitai\.com/models/(\d+)') {
        return $Matches[1]
    }
    return $null
}

function Get-CivitaiStats {
    param([string]$ModelId)

    $url = "https://civitai.com/api/v1/models/$ModelId"

    try {
        $response = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 30
        return @{
            Downloads = $response.stats.downloadCount
            Thumbs = $response.stats.thumbsUpCount
            Tips = $response.stats.tippedAmountCount
            Success = $true
        }
    }
    catch {
        Write-Warning "API error for model $ModelId : $_"
        return @{ Success = $false }
    }
}

function Get-ScoreRating {
    param([int]$Downloads)

    if ($Downloads -ge 5000) { return "$star1$star1$star1" }
    elseif ($Downloads -ge 2000) { return "$star1$star1" }
    elseif ($Downloads -ge 500) { return "$star1" }
    else { return "-" }
}

function Update-StatsInContent {
    param(
        [string]$Content,
        [int]$Downloads,
        [int]$Thumbs,
        [int]$Tips
    )

    $score = Get-ScoreRating -Downloads $Downloads

    # Update individual file format (Civitai Stats table)
    # | **Downloads** | XXX |
    $Content = $Content -replace '(\|\s*\*\*Downloads\*\*\s*\|)\s*[\d,\-]+\s*\|', "`$1 $Downloads |"

    # | **👍** | XXX |  (or **Rating**)
    $Content = $Content -replace '(\|\s*\*\*(?:' + $thumbs + '|Rating)\*\*\s*\|)\s*[\d,\-]+\s*\|', "`$1 $Thumbs |"

    # | **Tips** | XXX |
    $Content = $Content -replace '(\|\s*\*\*Tips\*\*\s*\|)\s*[\d,\-]+\s*\|', "`$1 $Tips |"

    # | **Score** | XXX |
    $Content = $Content -replace '(\|\s*\*\*Score\*\*\s*\|)\s*[^\|]+\|', "`$1 $score |"

    return $Content
}

function Update-IndexStats {
    param(
        [string]$IndexPath,
        [hashtable]$AllStats  # key = filename, value = stats hashtable
    )

    if (-not (Test-Path $IndexPath)) { return }

    $content = Get-Content $IndexPath -Raw -Encoding UTF8
    $updated = $false

    foreach ($entry in $AllStats.GetEnumerator()) {
        $filename = $entry.Key
        $s = $entry.Value

        # Pattern: | [Name](filename.md) | downloads | thumbs | tips | score |
        $pattern = "(\|\s*\[[^\]]+\]\($filename\)\s*\|)\s*[\d,\-]+\s*\|\s*[\d,\-]+\s*\|\s*[\d,\-]+\s*\|\s*[^\|]+\|"
        $score = Get-ScoreRating -Downloads $s.Downloads
        $replacement = "`$1 $($s.Downloads) | $($s.Thumbs) | $($s.Tips) | $score |"

        if ($content -match $pattern) {
            $content = $content -replace $pattern, $replacement
            $updated = $true
        }
    }

    if ($updated -and -not $DryRun) {
        Set-Content -Path $IndexPath -Value $content -NoNewline -Encoding UTF8
        if ($Verbose) { Write-Host "  Updated INDEX.md" -ForegroundColor Green }
    }
}

# Main execution
Write-Host "=== ComfyUI Documentation Stats Updater ===" -ForegroundColor Cyan
Write-Host "Docs path: $docsPath"
if ($DryRun) { Write-Host "DRY RUN MODE - no changes will be made" -ForegroundColor Yellow }
Write-Host ""

# Get files to process
$searchPath = if ($Category) { Join-Path $docsPath $Category } else { $docsPath }
$files = Get-ChildItem $searchPath -Recurse -Filter "*.md" | Where-Object { $_.Name -ne "INDEX.md" -and $_.Name -ne "CLAUDE.md" }

Write-Host "Found $($files.Count) documentation files to process"
Write-Host ""

# Group files by directory for INDEX.md updates
$filesByDir = @{}

foreach ($file in $files) {
    $stats.FilesProcessed++
    $content = Get-Content $file.FullName -Raw -Encoding UTF8

    # Get Civitai model ID
    $modelId = Get-CivitaiModelId -Content $content

    if (-not $modelId) {
        if ($Verbose) { Write-Host "[$($stats.FilesProcessed)/$($files.Count)] SKIP: $($file.Name) - no Civitai URL" -ForegroundColor Gray }
        $stats.FilesSkipped++
        continue
    }

    if ($Verbose) { Write-Host "[$($stats.FilesProcessed)/$($files.Count)] Fetching: $($file.Name) (model $modelId)..." -ForegroundColor White }

    # Fetch stats from API
    $apiStats = Get-CivitaiStats -ModelId $modelId

    if (-not $apiStats.Success) {
        $stats.ApiErrors++
        continue
    }

    # Update content
    $newContent = Update-StatsInContent -Content $content -Downloads $apiStats.Downloads -Thumbs $apiStats.Thumbs -Tips $apiStats.Tips

    if ($newContent -ne $content) {
        if (-not $DryRun) {
            Set-Content -Path $file.FullName -Value $newContent -NoNewline -Encoding UTF8
        }
        $stats.FilesUpdated++

        if ($Verbose) {
            Write-Host "  -> Downloads: $($apiStats.Downloads), $thumbs : $($apiStats.Thumbs), Tips: $($apiStats.Tips)" -ForegroundColor Green
        }

        # Store for INDEX.md update
        $dir = $file.DirectoryName
        if (-not $filesByDir.ContainsKey($dir)) {
            $filesByDir[$dir] = @{}
        }
        $filesByDir[$dir][$file.Name] = $apiStats
    }

    # Rate limiting delay
    Start-Sleep -Milliseconds $DelayMs
}

# Update INDEX.md files
Write-Host ""
Write-Host "Updating INDEX.md files..." -ForegroundColor Cyan

foreach ($dir in $filesByDir.Keys) {
    $indexPath = Join-Path $dir "INDEX.md"
    if (Test-Path $indexPath) {
        Update-IndexStats -IndexPath $indexPath -AllStats $filesByDir[$dir]
    }
}

# Summary
Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Files processed: $($stats.FilesProcessed)"
Write-Host "Files updated:   $($stats.FilesUpdated)" -ForegroundColor Green
Write-Host "Files skipped:   $($stats.FilesSkipped)" -ForegroundColor Gray
Write-Host "API errors:      $($stats.ApiErrors)" -ForegroundColor $(if ($stats.ApiErrors -gt 0) { "Red" } else { "Gray" })

# Update timestamp in CLAUDE.md
$claudePath = Join-Path $docsPath "CLAUDE.md"
if ((Test-Path $claudePath) -and -not $DryRun) {
    $today = Get-Date -Format "yyyy-MM-dd"
    $claudeContent = Get-Content $claudePath -Raw -Encoding UTF8
    $claudeContent = $claudeContent -replace '\*Last updated: \d{4}-\d{2}-\d{2}\*', "*Last updated: $today*"
    Set-Content -Path $claudePath -Value $claudeContent -NoNewline -Encoding UTF8
    Write-Host "Updated CLAUDE.md timestamp to $today" -ForegroundColor Green
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
