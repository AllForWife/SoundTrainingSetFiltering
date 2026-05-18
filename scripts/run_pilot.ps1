param(
    [int]$Limit = 100,
    [string]$Backend = "auto"
)

$ErrorActionPreference = "Stop"
python .\voice_filter.py run-all --limit $Limit --random --backend $Backend
Get-Content .\out\report.md
