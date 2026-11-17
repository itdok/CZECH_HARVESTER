$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir "GeminiVoiceAgent.lnk"

$Pythonw = (Get-Command pythonw.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)
if (-not $Pythonw) {
    $Candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\pythonw.exe"
    if (Test-Path $Candidate) {
        $Pythonw = $Candidate
    } else {
        throw "pythonw.exe not found in PATH or default Python 3.12 location."
    }
}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Pythonw
$Shortcut.Arguments = "`"$ProjectDir\run_agent.py`""
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.WindowStyle = 7  # Minimized / hidden
$Shortcut.Description = "Gemini Voice Agent for Telegram (CZECH_HARVESTER)"
$Shortcut.Save()

Write-Host "Autostart shortcut created at: $ShortcutPath"
Write-Host "Target: $Pythonw `"$ProjectDir\run_agent.py`""
