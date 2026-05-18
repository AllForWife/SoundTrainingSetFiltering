param(
    [string]$Python = "py -3.11",
    [string]$VenvPath = ".venv-uvr",
    [switch]$InstallAudioSeparator = $true
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $VenvPath)) {
    & py -3.11 -m venv $VenvPath
}

$pythonExe = Join-Path $VenvPath "Scripts\python.exe"
& $pythonExe -m pip install --upgrade pip setuptools wheel

if ($InstallAudioSeparator) {
    & $pythonExe -m pip install "audio-separator==0.44.1"
}

Write-Host "UVR python: $pythonExe"
Write-Host "Use: `$env:VOICE_FILTER_UVR_PYTHON = '$pythonExe'"
