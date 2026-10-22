$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$latexmk = "D:\texlive\2026\bin\windows\latexmk.exe"
if (-not (Test-Path $latexmk)) {
    throw "latexmk not found at $latexmk"
}

& $latexmk -xelatex main.tex
if ($LASTEXITCODE -ne 0) {
    throw "LaTeX build failed"
}

$out = Join-Path $PSScriptRoot "..\submission\ISY5002_Final_Report.pdf"
Copy-Item (Join-Path $PSScriptRoot "build\main.pdf") $out -Force
Write-Output "PDF -> $out"
