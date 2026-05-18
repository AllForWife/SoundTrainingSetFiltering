$ErrorActionPreference = "Stop"

python -m PyInstaller `
  --clean `
  --onefile `
  --windowed `
  --name VoiceFilterGUI `
  --paths "src" `
  --hidden-import "voice_filter_pipeline.pipeline" `
  --hidden-import "voice_filter_pipeline.audio" `
  --hidden-import "voice_filter_pipeline.features" `
  --hidden-import "voice_filter_pipeline.separation" `
  --hidden-import "numpy" `
  --hidden-import "soundfile" `
  gui_voice_filter.py

Write-Host "Built dist\VoiceFilterGUI.exe"
