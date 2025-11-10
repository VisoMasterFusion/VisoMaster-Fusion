@echo off
setlocal EnableDelayedExpansion

:: ===================================================================
::  VisoMaster Fusion Portable Launcher
:: ===================================================================

:: --- Step 0: Initial Path and Variable Setup ---
echo VisoMaster Fusion Portable Launcher
echo ===========================================
echo.

set "BASE_DIR=%~dp0"
set "PORTABLE_DIR=%BASE_DIR%portable-files"
set "PORTABLE_CFG=%BASE_DIR%portable.cfg"
set "REPO_URL=https://github.com/VisoMasterFusion/VisoMaster-Fusion.git"
for %%a in ("%REPO_URL%") do set "REPO_NAME=%%~na"
set "APP_DIR=%BASE_DIR%%REPO_NAME%"

:: Portable Dependency Paths
set "PYTHON_DIR=%PORTABLE_DIR%\python"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "UV_DIR=%PORTABLE_DIR%\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "GIT_DIR=%PORTABLE_DIR%\git"
set "GIT_EXE=%GIT_DIR%\bin\git.exe"
set "VENV_DIR=%PORTABLE_DIR%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "FFMPEG_EXTRACT_DIR=%BASE_DIR%dependencies"
set "FFMPEG_DIR_NAME=ffmpeg-7.1.1-essentials_build"
set "FFMPEG_BIN_PATH=%FFMPEG_EXTRACT_DIR%\%FFMPEG_DIR_NAME%\bin"

:: Download URLs and temp file paths
set "PYTHON_EMBED_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PYTHON_NUGET_URL=https://www.nuget.org/api/v2/package/python/3.11.9"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-pc-windows-msvc.zip"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.51.0.windows.1/PortableGit-2.51.0-64-bit.7z.exe"
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-essentials_build.zip"

set "PYTHON_ZIP=%PORTABLE_DIR%\python-embed.zip"
set "PYTHON_NUGET_ZIP=%PORTABLE_DIR%\python-nuget.zip"
set "UV_ZIP=%PORTABLE_DIR%\uv.zip"
set "GIT_ZIP=%PORTABLE_DIR%\PortableGit.exe"
set "FFMPEG_ZIP=%PORTABLE_DIR%\ffmpeg.zip"

:: --- LAUNCHER MODE PRE-CHECK (Fast path for existing installs) ---
set "LAUNCHER_ENABLED="
if exist "%PORTABLE_CFG%" (
  for /f "usebackq tokens=1,* delims== " %%A in ("%PORTABLE_CFG%") do (
    if /I "%%A"=="LAUNCHER_ENABLED" set "LAUNCHER_ENABLED=%%B"
  )
)
if "%LAUNCHER_ENABLED%"=="1" (
  if exist "%VENV_PYTHON%" (
    if exist "%APP_DIR%\.git" (
      echo Existing installation detected. Starting Launcher...
      echo.
      set "PYTHONPATH=%APP_DIR%"
      set "PATH=%FFMPEG_BIN_PATH%;%PATH%"
      "%VENV_PYTHON%" -m app.ui.launcher
      echo Launcher closed. Press any key to exit...
      pause >nul
      exit /b !ERRORLEVEL!
    )
  )
)

:: --- FULL SETUP / COMMAND-LINE MODE ---
echo Entering Full Setup / Command-Line Mode...
echo.

:: --- Step 1 & 2: Install Core Dependencies (Git, Python, UV) ---
if not exist "%PORTABLE_DIR%" mkdir "%PORTABLE_DIR%"
call :install_dependency "Git" "%GIT_EXE%" "%GIT_URL%" "%GIT_ZIP%" "%GIT_DIR%"
if not exist "%PYTHON_EXE%" (
    echo Installing Python...
    call :install_python
)
call :install_dependency "UV" "%UV_EXE%" "%UV_URL%" "%UV_ZIP%" "%UV_DIR%"

:: --- Step 3: Clone Repository (if it doesn't exist) ---
if not exist "%APP_DIR%\.git" (
    if exist "%APP_DIR%" (
        echo WARNING: %APP_DIR% exists but is not a Git repository. Cleaning folder...
        rmdir /s /q "%APP_DIR%"
    )
    echo Cloning repository...
    "%GIT_EXE%" clone "%REPO_URL%" "%APP_DIR%"
    if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to clone repository. && pause && exit /b 1)
)

:: --- Step 4: Determine and Set Branch ---
set "BRANCH="
if exist "%PORTABLE_CFG%" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%PORTABLE_CFG%") do if /I "%%a"=="BRANCH" set "BRANCH=%%b"
)
if not defined BRANCH (
    echo First run: Determining branch...
    if /I "%~1"=="dev" (
        set "BRANCH=dev"
        echo 'dev' argument found. Setting branch to dev.
    ) else (
        set "BRANCH=main"
        echo No argument found. Defaulting to main branch.
    )
    (echo BRANCH=!BRANCH!)>> "%PORTABLE_CFG%"
)
echo Using branch: !BRANCH!

:: --- Step 5: Git Checkout and Update ---
pushd "%APP_DIR%"
"%GIT_EXE%" checkout !BRANCH! >nul 2>&1
echo Checking for updates on branch '!BRANCH!'...
"%GIT_EXE%" fetch
for /f "tokens=*" %%i in ('"%GIT_EXE%" rev-parse HEAD') do set "LOCAL=%%i"
for /f "tokens=*" %%i in ('"%GIT_EXE%" rev-parse origin/!BRANCH!') do set "REMOTE=%%i"
set "NEEDS_INSTALL=false"
if not exist "%VENV_DIR%" set "NEEDS_INSTALL=true"

if "!LOCAL!" neq "!REMOTE!" (
    if "%LAUNCHER_ENABLED%"=="1" (
        echo Updates found. The launcher will handle the update process.
    ) else (
        echo Updates available.
        choice /c YN /m "Do you want to update? (Discards local changes) [Y/N] "
        if !ERRORLEVEL! equ 1 (
            echo Resetting to match remote...
            "%GIT_EXE%" reset --hard origin/!BRANCH!
            set "NEEDS_INSTALL=true"
        )
    )
) else (
    echo Repository is up to date.
)
popd

:: --- Step 6: Check for Launcher Self-Update ---
call :self_update_check

:: --- Step 7: Create Virtual Environment ---
if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment...
    "%UV_EXE%" venv "%VENV_DIR%" --python "%PYTHON_EXE%"
    if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to create venv. && pause && exit /b 1)
    set "NEEDS_INSTALL=true"
)

:: --- Step 8: Install Python Dependencies ---
set "REQ_FILE_NAME=requirements_cu129.txt"
set "REQUIREMENTS=%APP_DIR%\%REQ_FILE_NAME%"
if /I "!NEEDS_INSTALL!"=="true" (
    echo Installing dependencies from !REQ_FILE_NAME!...
    pushd "%APP_DIR%"
    "%UV_EXE%" pip install -r "!REQUIREMENTS!" --python "%VENV_PYTHON%"
    if !ERRORLEVEL! neq 0 ( echo ERROR: Dependency installation failed. && pause && exit /b 1 )
    popd
)

:: --- Step 9: Install FFmpeg ---
call :install_dependency "FFmpeg" "%FFMPEG_BIN_PATH%\ffmpeg.exe" "%FFMPEG_URL%" "%FFMPEG_ZIP%" "%FFMPEG_EXTRACT_DIR%"

:: --- Step 10: Download Models ---
set "DOWNLOAD_RUN=false"
if exist "%PORTABLE_CFG%" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%PORTABLE_CFG%") do if /I "%%a"=="DOWNLOAD_RUN" set "DOWNLOAD_RUN=%%b"
)
if /I "!NEEDS_INSTALL!"=="true" set "DOWNLOAD_RUN=false"
if /I "!DOWNLOAD_RUN!"=="false" (
    echo Running model downloader...
    pushd "%APP_DIR%"
    "%VENV_PYTHON%" "download_models.py"
    if !ERRORLEVEL! equ 0 (
        powershell -Command "(Get-Content -ErrorAction SilentlyContinue '%PORTABLE_CFG%') -replace 'DOWNLOAD_RUN=.*', 'DOWNLOAD_RUN=true' | Set-Content -ErrorAction SilentlyContinue '%PORTABLE_CFG%'"
    ) else (
        echo WARNING: Model download script failed.
    )
    popd
)

:: --- Final Launch ---
echo.
echo Starting main.py...
echo ========================================
pushd "%APP_DIR%"
set "PATH=%FFMPEG_BIN_PATH%;%PATH%"
"%VENV_PYTHON%" "main.py"
popd

echo.
echo Application closed. Press any key to exit...
pause >nul
endlocal
exit /b 0

:: ===================================================================
:: SUBROUTINES
:: ===================================================================

:self_update_check
    set "ROOT_BAT=%BASE_DIR%Start_Portable.bat"
    set "REPO_BAT=%APP_DIR%\Start_Portable.bat"

    if not exist "%REPO_BAT%" goto :eof

    fc /b "%ROOT_BAT%" "%REPO_BAT%" > nul
    if errorlevel 1 (
        echo A new version of the launcher script (Start_Portable.bat) is available.
        if "%LAUNCHER_ENABLED%"=="1" (
            echo Please use the launcher's Maintenance menu to update the script.
            goto :eof
        )

        choice /c YN /m "Do you want to update it now? [Y/N] "
        if !ERRORLEVEL! equ 1 (
            set "UPDATER_BAT=%PORTABLE_DIR%\update_start_portable.bat"
            (
                echo @echo off
                echo echo Waiting for main script to exit...
                echo timeout /t 2 /nobreak ^>nul
                echo echo Replacing Start_Portable.bat...
                echo copy /y "%REPO_BAT%" "%ROOT_BAT%"
                echo echo Update complete. Relaunching...
                echo start "" /d "%BASE_DIR%" "Start_Portable.bat"
                echo exit
            ) > "%UPDATER_BAT%"

            start "" cmd /c "%UPDATER_BAT%"
            exit /b 0
        )
    )
goto :eof

:install_dependency
    set "NAME=%~1"
    set "CHECK_FILE=%~2"
    set "URL=%~3"
    set "ZIP_FILE=%~4"
    set "EXTRACT_DIR=%~5"

    if exist "%CHECK_FILE%" (
        echo %NAME% already installed.
        goto :eof
    )

    echo Installing %NAME%...

    echo Downloading %NAME%...
    powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%URL%', '%ZIP_FILE%'); exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to download %NAME%.
        pause
        exit /b 1
    )

    echo Extracting %NAME%...
    mkdir "%EXTRACT_DIR%" >nul 2>&1
    if "%NAME%"=="Git" (
        "%ZIP_FILE%" -y -o"%EXTRACT_DIR%"
    ) else if "%NAME%"=="FFmpeg" (
        powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%EXTRACT_DIR%' -Force"
    ) else (
        powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%EXTRACT_DIR%' -Force"
    )

    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to extract %NAME%.
        del "%ZIP_FILE%"
        pause
        exit /b 1
    )
    del "%ZIP_FILE%"
goto :eof

:install_python
    echo Checking Windows version for Python installation...
    for /f "tokens=3 delims=." %%i in ('ver') do set WIN_BUILD=%%i

    if !WIN_BUILD! LSS 22000 (
        echo Windows 10 detected. Using full Python package.
        echo Downloading Python...
        powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%PYTHON_NUGET_URL%', '%PYTHON_NUGET_ZIP%'); exit 0 } catch { exit 1 }"
        if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to download Python. && pause && exit /b 1 )

        echo Extracting Python...
        set "TEMP_EXTRACT_DIR=%PORTABLE_DIR%\python_temp_extract"
        mkdir "!TEMP_EXTRACT_DIR!" >nul 2>&1
        powershell -Command "Expand-Archive -Path '%PYTHON_NUGET_ZIP%' -DestinationPath '!TEMP_EXTRACT_DIR!' -Force"
        move "!TEMP_EXTRACT_DIR!\tools" "%PYTHON_DIR%"
        rmdir /s /q "!TEMP_EXTRACT_DIR!"
        del "%PYTHON_NUGET_ZIP%"
    ) else (
        echo Windows 11 or newer detected. Using embeddable Python.
        echo Downloading Python...
        powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%PYTHON_EMBED_URL%', '%PYTHON_ZIP%'); exit 0 } catch { exit 1 }"
        if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to download Python. && pause && exit /b 1 )

        echo Extracting Python...
        mkdir "%PYTHON_DIR%" >nul 2>&1
        powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
        del "%ZIP_FILE%"

        set "PTH_FILE=%PYTHON_DIR%\python311._pth"
        if exist "!PTH_FILE!" (
            echo Enabling site packages in PTH file...
            powershell -Command "(Get-Content '!PTH_FILE!') -replace '#import site', 'import site' | Set-Content '!PTH_FILE!'"
        )
    )

    echo Installing pip...
    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://bootstrap.pypa.io/get-pip.py', '%PYTHON_DIR%\get-pip.py')"
    "%PYTHON_EXE%" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
    del "%PYTHON_DIR%\get-pip.py"
goto :eof
