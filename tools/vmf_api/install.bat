@echo off
setlocal
:: VisoMaster Fusion API / MCP - installation (idempotent)
:: Only needs re-running if portable-files\python is reinstalled.
set "VMF_DIR=%~dp0"
set "PY=%VMF_DIR%..\portable-files\python\python.exe"
:: inside the repository: VisoMaster-Fusion\tools\vmf_api
if not exist "%PY%" set "PY=%VMF_DIR%..\..\..\portable-files\python\python.exe"
if not exist "%PY%" goto no_python
"%PY%" -I "%VMF_DIR%install_helper.py" install
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%

:no_python
:: Outside an if ( ) block: a path with ")" (e.g. "Program Files (x86)") would end the block early.
echo [ERROR] Portable Python not found: %PY%
echo vmf_api must sit next to portable-files and VisoMaster-Fusion, or in VisoMaster-Fusion\tools.
pause
exit /b 1
