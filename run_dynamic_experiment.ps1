$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$Python = "C:\Users\zyx20\anaconda3\envs\myml\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "myml Python not found: $Python"
}

function Invoke-Stage {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Host "=== $Name ==="
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

$NormalizationStats = Join-Path $PSScriptRoot "trainresult\dynamic_observation_normalizer.json"
if (-not (Test-Path -LiteralPath $NormalizationStats -PathType Leaf)) {
    Invoke-Stage "Sample dynamic observation normalization" @(
        "collect_dynamic_observation_stats.py"
    )
}
else {
    Write-Host "=== Reuse dynamic observation normalization ==="
}

Invoke-Stage "Train dynamic PPO" @(
    "train.py",
    "--algo", "ppo"
)

Invoke-Stage "Train dynamic DQN" @(
    "train.py",
    "--algo", "dynamic-dqn"
)

Invoke-Stage "Evaluate dynamic policies" @(
    "eval.py",
    "--dynamic"
)

Invoke-Stage "Visualize dynamic policies" @(
    "visualize_dynamic.py"
)

Write-Host "=== Dynamic experiment completed ==="
