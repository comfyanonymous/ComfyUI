$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $DesktopPath "ComfyUI.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "C:\Repos\ComfyUI\run_comfyui.bat"
$Shortcut.WorkingDirectory = "C:\Repos\ComfyUI"
$Shortcut.Description = "Run ComfyUI from repository"
$Shortcut.Save()
Write-Host "Shortcut created at: $ShortcutPath"




