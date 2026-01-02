$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("C:\Users\spoko\Desktop\ComfyUI.lnk")
$Shortcut.TargetPath = "C:\Users\spoko\www\ai\ComfyUI\start_comfyui.bat"
$Shortcut.WorkingDirectory = "C:\Users\spoko\www\ai\ComfyUI"
$Shortcut.Description = "Start ComfyUI with Flux"
$Shortcut.Save()
Write-Host "Shortcut created on Desktop!"
