@echo off
setlocal
:: Removes vmf_api_hook.pth and vmf_api_bootstrap.py from the portable Python.
set "VMF_DIR=%~dp0"
set "PY=%VMF_DIR%..\portable-files\python\python.exe"
:: inside the repository: VisoMaster-Fusion\tools\vmf_api
if not exist "%PY%" set "PY=%VMF_DIR%..\..\..\portable-files\python\python.exe"
if not exist "%PY%" goto no_python
"%PY%" -I "%VMF_DIR%install_helper.py" uninstall
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%

:no_python
:: Outside an if ( ) block: a path with ")" (e.g. "Program Files (x86)") would end the block early.
echo Portable Python not found: nothing to remove.
pause
exit /b 0
