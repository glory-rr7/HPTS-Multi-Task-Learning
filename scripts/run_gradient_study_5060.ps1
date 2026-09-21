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

python .\run_gradient_study.py `
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
