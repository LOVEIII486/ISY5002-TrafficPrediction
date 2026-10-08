# 补跑自研模型缺失的种子。已有 run.json 的组合会自动跳过，可重复执行。
#
#   pwsh -NoProfile -File scripts\run_missing_seeds.ps1
#
# 日志追加到 results\forecast\resume_seeds.log

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $env:USERPROFILE 'miniconda3\envs\isy5002-project\python.exe'
$Log = 'results\forecast\resume_seeds.log'

$Targets = @('traffic_flow', 'traffic_occupancy', 'traffic_speed')
$Seeds = @(0, 1, 2, 3, 4)

New-Item -ItemType Directory -Force -Path 'results\forecast' | Out-Null
"=== 补跑缺失种子 $(Get-Date -Format s) ===" | Out-File -Append -Encoding utf8 $Log

$Ran = 0
foreach ($target in $Targets) {
    foreach ($seed in $Seeds) {
        $done = "results\forecast\PEMSD8\$target\MultiHeadSTGCN\seed$seed\run.json"
        if (Test-Path $done) { continue }

        "--- $target / seed $seed ---" | Out-File -Append -Encoding utf8 $Log
        Write-Host "--- $target / seed $seed ---"
        & $Python -u -m src.forecast.run_benchmark --dataset PEMSD8 --targets $target `
            --models MultiHeadSTGCN --seeds $seed *>> $Log
        "退出码 $LASTEXITCODE" | Out-File -Append -Encoding utf8 $Log
        $Ran++
    }
}

"=== 完成，补跑 $Ran 次 $(Get-Date -Format s) ===" | Out-File -Append -Encoding utf8 $Log
Write-Host "完成，补跑 $Ran 次"
