param(
    [string]$DatasetRoot = 'D:\gaorui\SignaTR6K',
    [string]$InitialCheckpoint = '.\data\train\Release_20260921_095504\models\0.pth',
    [int]$Epochs = 50,
    [int]$BatchSize = 4,
    [int]$Workers = 4
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

# A nested PowerShell can lose Conda's PATH ordering and can even inherit a
# stale base CONDA_PREFIX. Build candidates from the active environment name,
# then select the first interpreter that can actually import PyTorch.
$pythonCandidates = [System.Collections.Generic.List[string]]::new()
if ($env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne 'base') {
    $pythonCandidates.Add((Join-Path $env:USERPROFILE ".conda\envs\$($env:CONDA_DEFAULT_ENV)\python.exe"))
    if ($env:CONDA_EXE) {
        $condaRoot = Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE)
        $pythonCandidates.Add((Join-Path $condaRoot "envs\$($env:CONDA_DEFAULT_ENV)\python.exe"))
    }
}
if ($env:CONDA_PREFIX) {
    $pythonCandidates.Add((Join-Path $env:CONDA_PREFIX 'python.exe'))
}
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonCandidates.Add($pythonCommand.Source)
}

$pythonExecutable = $null
foreach ($candidate in ($pythonCandidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        continue
    }
    & $candidate -c "import torch" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonExecutable = $candidate
        break
    }
}
if (-not $pythonExecutable) {
    throw "No Python with PyTorch was found. Candidates: $($pythonCandidates -join ', ')"
}

$trainCandidates = @(
    (Join-Path $DatasetRoot 'train'),
    (Join-Path $DatasetRoot 'training')
)
$trainPath = $trainCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | Select-Object -First 1
$validationPath = Join-Path $DatasetRoot 'validation'

if (-not $trainPath) {
    throw "Training directory not found. Expected one of: $($trainCandidates -join ', ')"
}
if (-not (Test-Path -LiteralPath $validationPath -PathType Container)) {
    throw "Validation directory not found: $validationPath"
}
if (-not (Test-Path -LiteralPath $InitialCheckpoint -PathType Leaf)) {
    throw "Initial checkpoint not found: $InitialCheckpoint"
}

$env:PYTHONUNBUFFERED = '1'
Write-Host "Training data: $trainPath"
Write-Host "Validation data: $validationPath"
Write-Host "Common initialization: $InitialCheckpoint"
Write-Host 'Methods: baseline, balance, pcgrad, balance_pcgrad'
Write-Host "Python executable: $pythonExecutable"

& $pythonExecutable -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if ($LASTEXITCODE -ne 0) {
    throw "The selected Python environment cannot import PyTorch: $pythonExecutable"
}

& $pythonExecutable .\run_gradient_study.py `
    --train-path $trainPath `
    --validation-path $validationPath `
    --initial-checkpoint $InitialCheckpoint `
    --model Release `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --validation-batch-size $BatchSize `
    --workers $Workers `
    --validation-workers $Workers `
    --learning-rate 0.0002 `
    --save-every 5 `
    --diagnostic-interval 10 `
    --output-root .\data

if ($LASTEXITCODE -ne 0) {
    throw "Gradient study failed with exit code $LASTEXITCODE"
}
