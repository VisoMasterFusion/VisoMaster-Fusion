@echo off
setlocal EnableDelayedExpansion

:: ===================================================================
::  VisoMaster Fusion Portable Launcher
:: ===================================================================

:: --- LAUNCHER INTEGRATION (pre-check) ---
set "BASE_DIR=%~dp0"
set "APP_PYTHON=%BASE_DIR%portable-files\python\python.exe"
set "GIT_DIR_PRESENT=%BASE_DIR%VisoMaster-Fusion\.git"
set "PORTABLE_CFG=%BASE_DIR%portable.cfg"
set "FFMPEG_EXTRACT_DIR=%BASE_DIR%portable-files"
set "FFMPEG_DIR_NAME=ffmpeg-7.1.1-essentials_build"
set "FFMPEG_BIN_PATH=%FFMPEG_EXTRACT_DIR%\%FFMPEG_DIR_NAME%\bin"

set "LAUNCHER_ENABLED="
if exist "%PORTABLE_CFG%" (
  for /f "usebackq tokens=1,* delims== " %%A in ("%PORTABLE_CFG%") do (
    if /I "%%A"=="LAUNCHER_ENABLED" set "LAUNCHER_ENABLED=%%B"
  )
)

if "%LAUNCHER_ENABLED%"=="1" (
  if exist "%APP_PYTHON%" (
    if exist "%GIT_DIR_PRESENT%" (
      echo Existing installation detected. Launching VisoMaster Fusion Launcher...
      pushd "%BASE_DIR%VisoMaster-Fusion"
      ::set "PYTHONPATH=%BASE_DIR%VisoMaster-Fusion"
      set "PATH=%FFMPEG_BIN_PATH%;%PATH%"
      "%APP_PYTHON%" -m app.ui.launcher
      popd
      exit /b !ERRORLEVEL!
    )
  )
)

:: --- DEVELOPER SETUP LOGIC (Adapted from working old script) ---
echo Entering Full Setup / Command-Line Mode...
echo.

:: --- Basic Setup ---
set "REPO_URL=https://github.com/VisoMasterFusion/VisoMaster-Fusion.git"
for %%a in ("%REPO_URL%") do set "REPO_NAME=%%~na"
set "APP_DIR=%BASE_DIR%%REPO_NAME%"

:: Define portable tool paths
set "PORTABLE_DIR=%BASE_DIR%portable-files"
set "PYTHON_DIR=%PORTABLE_DIR%\python"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "UV_DIR=%PORTABLE_DIR%\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "GIT_DIR=%PORTABLE_DIR%\git"
set "GIT_EXE=%GIT_DIR%\bin\git.exe"

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

set "CONFIG_FILE=%BASE_DIR%portable.cfg"
set "NEEDS_INSTALL=false"

if not exist "%PORTABLE_DIR%" mkdir "%PORTABLE_DIR%"

:: --- Step 1: Set up portable Git ---
if not exist "%GIT_EXE%" (
    echo Downloading PortableGit...
    powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%GIT_URL%', '%GIT_ZIP%'); exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to download PortableGit. && pause && exit /b 1 )
    echo Extracting PortableGit...
    mkdir "%GIT_DIR%" >nul 2>&1
    "%GIT_ZIP%" -y -o"%GIT_DIR%"
    if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to extract PortableGit. && pause && exit /b 1 )
    del "%GIT_ZIP%"
)

:: --- Step 2: Determine Branch ---
set "BRANCH="
if exist "%CONFIG_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%CONFIG_FILE%") do if /I "%%a"=="BRANCH" set "BRANCH=%%b"
)
if not defined BRANCH (
    echo First run: Determining branch...
    if /I "%~1"=="dev" (
        set "BRANCH=dev"
        echo 'dev' argument found. Setting branch to dev.
    ) else (
        set "BRANCH=main"
        echo No argument provided. Defaulting to main branch.
    )
    (echo BRANCH=!BRANCH!)>> "%CONFIG_FILE%"
)
echo Using branch: !BRANCH!


:: --- Step 3: Clone or update repository ---
if not exist "%APP_DIR%\.git" (
    if exist "%APP_DIR%" (
        echo WARNING: %APP_DIR% exists but is not a git repo. Cleaning folder...
        rmdir /s /q "%APP_DIR%"
    )
    echo Cloning repository on branch '%BRANCH%'...
    "%GIT_EXE%" clone --branch "%BRANCH%" "%REPO_URL%" "%APP_DIR%"
    if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to clone repository. && pause && exit /b 1 )
    set "NEEDS_INSTALL=true"
) else (
    echo Repository exists. Checking for updates...

    :: Clear any git environment variables that might interfere
    set "GIT_DIR="
    set "GIT_WORK_TREE="

    pushd "%APP_DIR%"

    "%GIT_EXE%" checkout %BRANCH%
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to checkout branch.
        popd
        exit /b 1
    )

    "%GIT_EXE%" fetch
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to fetch updates.
        popd
        exit /b 1
    )

    for /f "tokens=*" %%i in ('"%GIT_EXE%" rev-parse HEAD') do set "LOCAL=%%i"
    for /f "tokens=*" %%i in ('"%GIT_EXE%" rev-parse origin/%BRANCH%') do set "REMOTE=%%i"

    if "!LOCAL!" neq "!REMOTE!" (
        echo Updates available on branch %BRANCH%.
        choice /c YN /m "Do you want to update? (This will discard local changes) (Y/N) "
        if !ERRORLEVEL! equ 1 (
            echo Resetting local repository to match remote...
            "%GIT_EXE%" reset --hard origin/%BRANCH%
            if !ERRORLEVEL! neq 0 (
                echo ERROR: Failed to reset repository.
                popd
                exit /b 1
            )
            echo Repository updated.
            set "NEEDS_INSTALL=true"
            set "DOWNLOAD_RUN=false"
            popd
            powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'DOWNLOAD_RUN=.*', 'DOWNLOAD_RUN=false' | Set-Content '%CONFIG_FILE%'"

            :: SELF-UPDATE CHECK
            call :self_update_check
        ) else (
            popd
        )
    ) else (
        echo Repository is up to date.
        popd
    )
)

:: --- Step 4: Self-update check ---
call :self_update_check

:: --- Step 5: Install Python, UV ---
if not exist "%PYTHON_EXE%" (
    call :install_python
    set "NEEDS_INSTALL=true"
)
call :install_dependency "UV" "%UV_EXE%" "%UV_URL%" "%UV_ZIP%" "%UV_DIR%"

:: --- Ensure dependencies reinstall on fresh setup ---
if /I "!NEEDS_INSTALL!"=="false" (
    if not exist "%PYTHON_DIR%\Lib\site-packages\PySide6" (
        echo Fresh Python environment detected - forcing dependency install...
        set "NEEDS_INSTALL=true"
    )
)

:: --- Step 6: Install dependencies ---
set "REQUIREMENTS=%APP_DIR%\requirements_cu129.txt"
if /I "!NEEDS_INSTALL!"=="true" (
    echo Installing dependencies...
    pushd "%APP_DIR%"
    "%UV_EXE%" pip install -r "!REQUIREMENTS!" --python "%APP_PYTHON%"
    if !ERRORLEVEL! neq 0 ( echo ERROR: Dependency installation failed. && pause && exit /b 1 )
    popd
)

:: --- Step 7: Install FFmpeg ---
call :install_dependency "FFmpeg" "%FFMPEG_BIN_PATH%\ffmpeg.exe" "%FFMPEG_URL%" "%FFMPEG_ZIP%" "%FFMPEG_EXTRACT_DIR%"

:: --- Step 8: Download models ---
set "DOWNLOAD_RUN=false"
if exist "%CONFIG_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%CONFIG_FILE%") do if /I "%%a"=="DOWNLOAD_RUN" set "DOWNLOAD_RUN=%%b"
)
if /I "!NEEDS_INSTALL!"=="true" set "DOWNLOAD_RUN=false"
if /I "!DOWNLOAD_RUN!"=="false" (

    echo Running model downloader next. If you already downloaded the models, copy them now to the VisoMaster-Fusion\model_assets directory and then press enter to avoid re-download.
    pause
    echo Running model downloader...
    pushd "%APP_DIR%"
    ::set "PYTHONPATH=%APP_DIR%"
    "%APP_PYTHON%" "download_models.py"
    if !ERRORLEVEL! equ 0 (
        powershell -Command "(Get-Content -ErrorAction SilentlyContinue '%CONFIG_FILE%') -replace 'DOWNLOAD_RUN=.*', 'DOWNLOAD_RUN=true' | Set-Content -ErrorAction SilentlyContinue '%CONFIG_FILE%'"
    )
    popd
)

:: --- Ensure LAUNCHER_ENABLED key exists ---
set "FOUND_KEY=false"
if exist "%CONFIG_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%CONFIG_FILE%") do if /I "%%~A"=="LAUNCHER_ENABLED" set "FOUND_KEY=true"
)
if /I "!FOUND_KEY!"=="false" (
    echo.
    echo Setting LAUNCHER_ENABLED=1 in portable.cfg...
    (echo LAUNCHER_ENABLED=1)>> "%CONFIG_FILE%"
    set LAUNCHER_ENABLED=1
)

:: --- Step 9: Run main application ---
if "%LAUNCHER_ENABLED%"=="1" (
    set "PATH=%FFMPEG_BIN_PATH%;%PATH%"
    "%APP_PYTHON%" -m app.ui.launcher
    exit /b !ERRORLEVEL!
) else (
    echo.
    echo Starting main.py...
    echo ========================================
    pushd "%APP_DIR%"
    set "PYTHONPATH=%APP_DIR%"
    set "PATH=%FFMPEG_BIN_PATH%;%GIT_DIR%\bin;%PATH%"
    "%APP_PYTHON%" "main.py"
    popd

    echo.
    echo Application closed. Press any key to exit...
    pause >nul
    endlocal
    exit /b 0
)

:: ===================================================================
:: SUBROUTINES
:: ===================================================================

:self_update_check
    set "ROOT_BAT=%BASE_DIR%Start_Portable.bat"
    set "REPO_BAT=%APP_DIR%\Start_Portable.bat"

    if not exist "%REPO_BAT%" goto :eof

    fc /b "%ROOT_BAT%" "%REPO_BAT%" > nul
    if errorlevel 1 (
        echo A new version of the launcher script Start_Portable.bat is available.
        if "%LAUNCHER_ENABLED%"=="1" (
            echo Please restart by running Start_Portable.bat to apply the update.
            pause
            exit /b 0
        )

        choice /c YN /m "Do you want to update it now? "
        if !ERRORLEVEL! equ 1 (
            set "UPDATER_BAT=%PORTABLE_DIR%\update_start_portable.bat"
            (
                echo @echo off
                echo echo Waiting for main script to exit...
                echo timeout /t 2 /nobreak ^>nul
                echo echo Replacing Start_Portable.bat...
                echo copy /y "%REPO_BAT%" "%ROOT_BAT%"
                echo if errorlevel 1 ^(
                echo     echo ERROR: Failed to update launcher script.
                echo     pause
                echo     exit /b 1
                echo ^)
                echo echo Update complete. Relaunching...
                echo start "" /d "%BASE_DIR%" "%ROOT_BAT%"
                echo del "%%~f0"
            ) > "!UPDATER_BAT!"

            echo Launching updater and exiting...
            start "" cmd /c "!UPDATER_BAT!"
            exit
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
    if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to download %NAME%. && pause && exit /b 1 )

    echo Extracting %NAME%...
    mkdir "%EXTRACT_DIR%" >nul 2>&1
    if "%NAME%"=="Git" (
        "%ZIP_FILE%" -y -o"%EXTRACT_DIR%"
    ) else if "%NAME%"=="FFmpeg" (
        powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%EXTRACT_DIR%' -Force"
    ) else (
        powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%EXTRACT_DIR%' -Force"
    )
    if !ERRORLEVEL! neq 0 ( echo ERROR: Failed to extract %NAME%. && del "%ZIP_FILE%" && pause && exit /b 1 )
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
        del "%PYTHON_ZIP%"
        set "PTH_FILE=%PYTHON_DIR%\python311._pth"
        if exist "!PTH_FILE!" (
            echo Enabling site packages in PTH file...
            powershell -Command "(Get-Content '!PTH_FILE!') -replace '#import site', 'import site' | Set-Content '!PTH_FILE!'"
        )
        :: --- Add repo path to python311._pth for app imports ---
        if exist "!PTH_FILE!" (
            echo Adding portable repo path to python311._pth...
            powershell -Command ^
                "$pth = Get-Content '!PTH_FILE!';" ^
                "if (-not ($pth -match '\.\./\.\./VisoMaster-Fusion')) {" ^
                "    Add-Content '!PTH_FILE!' ' ../../VisoMaster-Fusion';" ^
                "}"
        )
    )

    echo Installing pip...
    powershell -Command "(New-Object Net.WebClient).DownloadFile('https://bootstrap.pypa.io/get-pip.py', '%PYTHON_DIR%\get-pip.py')"
    "%PYTHON_EXE%" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
    del "%PYTHON_DIR%\get-pip.py"
goto :eof
