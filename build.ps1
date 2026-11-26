$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& $taskPython -m pip install -r requirements.txt pyinstaller
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $taskPython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $taskPython -m PyInstaller --noconfirm --windowed --onedir --distpath dist-auto-chase --name SlimeAssistant run.py
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.txt' | Where-Object Name -ne 'requirements.txt' | Copy-Item -Destination '.\dist-auto-chase\SlimeAssistant'
Compress-Archive -Path '.\dist-auto-chase\SlimeAssistant' -DestinationPath '.\dist-auto-chase\SlimeAssistant-AutoChase.zip' -Force
