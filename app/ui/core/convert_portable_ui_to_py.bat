@echo off
setlocal enabledelayedexpansion

:: Define relative paths
set "UI_FILE=app\ui\core\MainWindow.ui"
set "PY_FILE=app\ui\core\main_window.py"
set "QRC_FILE=app\ui\core\media.qrc"
set "RCC_PY_FILE=app\ui\core\media_rc.py"

:: Find Project Root (Embedded)
call :findRoot
if "%R%"=="" exit /b 1
set "UIC=%R%\dependencies\Python\Scripts\pyside6-uic.exe"
set "RCC=%R%\dependencies\Python\Scripts\pyside6-rcc.exe"

:: Run PySide6 commands
"%UIC%" "%UI_FILE%" -o "%PY_FILE%"
"%RCC%" "%QRC_FILE%" -o "%RCC_PY_FILE%"

:: Define search and replace strings
set "searchString=import media_rc"
set "replaceString=from app.ui.core import media_rc"

:: Create a temporary file
set "tempFile=%PY_FILE%.tmp"

:: Process the file
(for /f "usebackq delims=" %%A in ("%PY_FILE%") do (
    set "line=%%A"
    if "!line!"=="%searchString%" (
        echo %replaceString%
    ) else (
        echo !line!
    )
)) > "%tempFile%"

:: Replace the original file with the temporary file
move /y "%tempFile%" "%PY_FILE%" > nul

exit /b 0

:findRoot
set "R="
set "F=main.py"
if exist "%CD%\%F%" (set "R=%CD%"&exit /b 0)
if "%CD:~2%"=="" (echo ERROR: No root found.&exit /b 1)
cd ..&goto findRoot