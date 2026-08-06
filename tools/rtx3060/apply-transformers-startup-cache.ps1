$ErrorActionPreference = 'Stop'

$comfyRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$target = Join-Path $comfyRoot 'venv/Lib/site-packages/transformers/utils/import_utils.py'
$python = Join-Path $comfyRoot 'venv/Scripts/python.exe'

if (-not (Test-Path -LiteralPath $target)) {
    throw "Transformers import utility not found: $target"
}

$text = [IO.File]::ReadAllText($target).Replace("`r`n", "`n")
$changed = $false

if (-not $text.Contains('class _DirectDistributionMapping:')) {
    $oldMapping = 'PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()'
    if (-not $text.Contains($oldMapping)) {
        throw 'Unsupported transformers version: package mapping marker was not found.'
    }

    $newMapping = @'
class _DirectDistributionMapping:
    """Avoid the expensive full-environment scan done by packages_distributions()."""

    _ALIASES = {
        "PIL": ["pillow"],
        "OpenSSL": ["pyopenssl"],
        "bs4": ["beautifulsoup4"],
        "cv2": ["opencv-python", "opencv-python-headless"],
        "dateutil": ["python-dateutil"],
        "sklearn": ["scikit-learn"],
        "yaml": ["pyyaml"],
    }

    def __getitem__(self, package_name: str) -> list[str]:
        return self._ALIASES.get(package_name, [package_name.replace("_", "-")])

    def get(self, package_name: str, default=None):
        return self._ALIASES.get(package_name, [package_name.replace("_", "-")]) or default


PACKAGE_DISTRIBUTION_MAPPING = _DirectDistributionMapping()
'@
    $text = $text.Replace($oldMapping, $newMapping.TrimEnd())
    $changed = $true
}

if (-not $text.Contains("import pickle`n")) {
    $oldImports = "import os`nimport re"
    if (-not $text.Contains($oldImports)) {
        throw 'Unsupported transformers version: import marker was not found.'
    }
    $text = $text.Replace($oldImports, "import os`nimport pickle`nimport re")
    $changed = $true
}

if (-not $text.Contains('.import_structure_cache.pkl')) {
    $oldBody = @'
    import_structure = create_import_structure_from_path(module_path)
    spread_dict = spread_import_structure(import_structure)

    if prefix is None:
        return spread_dict
    else:
        spread_dict = {k: {f"{prefix}.{kk}": vv for kk, vv in v.items()} for k, v in spread_dict.items()}
        return spread_dict
'@
    $newBody = @'
    module_path = os.fspath(module_path)
    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".import_structure_cache.pkl")
    cache_key = (os.path.normcase(os.path.abspath(module_path)), prefix)

    try:
        with open(cache_path, "rb") as cache_file:
            cache = pickle.load(cache_file)
        if cache.get("key") == cache_key:
            return cache["value"]
    except (FileNotFoundError, OSError, EOFError, pickle.PickleError, AttributeError, TypeError):
        pass

    import_structure = create_import_structure_from_path(module_path)
    spread_dict = spread_import_structure(import_structure)

    if prefix is not None:
        spread_dict = {k: {f"{prefix}.{kk}": vv for kk, vv in v.items()} for k, v in spread_dict.items()}

    try:
        temporary_cache_path = cache_path + ".tmp"
        with open(temporary_cache_path, "wb") as cache_file:
            pickle.dump({"key": cache_key, "value": spread_dict}, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_cache_path, cache_path)
    except OSError:
        pass

    return spread_dict
'@
    if (-not $text.Contains($oldBody.TrimEnd())) {
        throw 'Unsupported transformers version: import-structure marker was not found.'
    }
    $text = $text.Replace($oldBody.TrimEnd(), $newBody.TrimEnd())
    $changed = $true
}

if ($changed) {
    [IO.File]::WriteAllText($target, $text, [Text.UTF8Encoding]::new($false))
    Write-Host 'Applied transformers startup-cache patch.' -ForegroundColor Green
} else {
    Write-Host 'Transformers startup-cache patch is already applied.' -ForegroundColor DarkGreen
}

$cache = Join-Path (Split-Path (Split-Path $target -Parent) -Parent) '.import_structure_cache.pkl'
if (-not (Test-Path -LiteralPath $cache)) {
    Write-Host 'Building transformers import cache. The first run can take about two minutes on this drive...' -ForegroundColor Yellow
    & $python -X utf8 -c "import transformers; print('transformers=' + transformers.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "Transformers cache build failed with exit code $LASTEXITCODE"
    }
}
