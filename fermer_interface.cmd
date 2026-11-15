@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0fermer_interface.ps1"
if errorlevel 1 pause
