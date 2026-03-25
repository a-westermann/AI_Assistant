$ErrorActionPreference = "SilentlyContinue"

# Stop any previous Galadrial API uvicorn process started from python.exe
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -eq "python.exe" -and
    $_.CommandLine -match "uvicorn\s+api_server:app"
  } |
  ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force
  }

Start-Sleep -Seconds 1

Set-Location "H:\Coding\AI_assistant"
& "H:\Coding\AI_assistant\venv\Scripts\python.exe" -m uvicorn api_server:app --host 0.0.0.0 --port 8000
