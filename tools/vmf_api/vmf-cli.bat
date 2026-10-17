@echo off
rem vmf-cli shortcut: runs vmf_cli.py with this folder's venv, from any folder.
rem Example: vmf-cli status    (add this folder to PATH to type it anywhere)
"%~dp0.venv\Scripts\python.exe" "%~dp0vmf_cli.py" %*
exit /b %ERRORLEVEL%
