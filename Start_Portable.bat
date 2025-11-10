@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  VisoMaster Fusion Portable Launcher
::  - First run: performs full setup and writes LAUNCHER_ENABLED=1
::  - Subsequent runs: launches PySide6 GUI launcher
::  - If key missing: automatically repairs portable.cfg
:: ============================================================

:: --- LAUNCHER INTEGRATION (pre-check) ---
:: Minimal pre-check — don’t redefine dev’s path variables
set "BASE_DIR=%~dp0"
set "VENV_PYTHON=%BASE_DIR%portable-files\venv\Scripts\python.exe"
set "GIT_DIR_PRESENT=%BASE_DIR%VisoMaster-Fusion\.git"
set "PORTABLE_CFG=%BASE_DIR%portable.cfg"

:: Read optional toggle from portable.cfg (no defaults here)
set "LAUNCHER_ENABLED="
if exist "%PORTABLE_CFG%" (
  for /f "usebackq tokens=1,* delims== " %%A in ("%PORTABLE_CFG%") do (
    if /I "%%A"=="LAUNCHER_ENABLED" set "LAUNCHER_ENABLED=%%B"
  )
)

:: Detect if this is the first run (no portable.cfg)
set "FIRST_RUN=false"
if not exist "%PORTABLE_CFG%" set "FIRST_RUN=true"


:: If GUI enabled and an installation already exists, start the launcher.
if "%LAUNCHER_ENABLED%"=="1" (
  if exist "%VENV_PYTHON%" (
    if exist "%GIT_DIR_PRESENT%" (
      echo Existing installation detected.
      echo Launching VisoMaster Fusion Launcher...
      echo ========================================
      echo Reminder:
      echo If you ever need to re-run the full setup,
      echo delete "portable.cfg" or rename "VisoMaster-Fusion".
      echo.

      :: Make sure Python knows where to find 'app'
      set "PYTHONPATH=%BASE_DIR%VisoMaster-Fusion"

      "%VENV_PYTHON%" -m app.ui.launcher
      set "LAUNCHER_EXIT_CODE=!ERRORLEVEL!"

      echo.
      echo Launcher closed. (Exit code: !LAUNCHER_EXIT_CODE!)
      echo Press any key to exit...
      pause >nul
      exit /b !LAUNCHER_EXIT_CODE!
    )
  )
)

:: If GUI disabled (LAUNCHER_ENABLED=0) or missing files, fall through
:: to the original developer setup logic below unchanged.


:: ============================================================
::  Original Developer Setup Logic (unchanged)
:: ============================================================

:: --- Basic Setup ---
:: Define repo details
set "REPO_URL=https://github.com/VisoMasterFusion/VisoMaster-Fusion.git"
set "BRANCH=dev"

:: Extract repo name from URL
for %%a in ("%REPO_URL%") do set "REPO_NAME=%%~na"

:: Define paths
set "BASE_DIR=%~dp0"
set "PORTABLE_DIR=%BASE_DIR%portable-files"
set "APP_DIR=%BASE_DIR%%REPO_NAME%"
set "PYTHON_DIR=%PORTABLE_DIR%\python"
set "UV_DIR=%PORTABLE_DIR%\uv"
set "GIT_DIR=%PORTABLE_DIR%\git"
set "GIT_BIN=%PORTABLE_DIR%\git\bin\git.exe"
set "VENV_DIR=%PORTABLE_DIR%\venv"
set "UV_CACHE_DIR=%PORTABLE_DIR%\uv-cache"
set "PYTHON_EMBED_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PYTHON_NUGET_URL=https://www.nuget.org/api/v2/package/python/3.11.9"
set "PYTHON_ZIP=%PORTABLE_DIR%\python-embed.zip"
set "PYTHON_NUGET_ZIP=%PORTABLE_DIR%\python-nuget.zip"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP=%PORTABLE_DIR%\uv.zip"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.51.0.windows.1/PortableGit-2.51.0-64-bit.7z.exe"
set "GIT_ZIP=%PORTABLE_DIR%\PortableGit.exe"
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-essentials_build.zip"
set "FFMPEG_ZIP=%PORTABLE_DIR%\ffmpeg.zip"
set "FFMPEG_EXTRACT_DIR=%BASE_DIR%dependencies"
set "FFMPEG_DIR_NAME=ffmpeg-7.1.1-essentials_build"
set "FFMPEG_PATH_VAR=%FFMPEG_EXTRACT_DIR%\%FFMPEG_DIR_NAME%\bin"

set "OLD_PATH=%PATH%"
set "PATH=%GIT_DIR%\bin;%PATH%"

set "CONFIG_FILE=%BASE_DIR%portable.cfg"
set "DOWNLOAD_PY=download_models.py"
set "MAIN_PY=main.py"

:: --- Step 0: User Configuration ---
if exist "%CONFIG_FILE%" (
    echo Loading configuration from portable.cfg...
    for /f "usebackq tokens=1,* delims==" %%a in ("%CONFIG_FILE%") do set "%%a=%%b"
    goto :ConfigLoaded
)

:: First time setup
set "REQ_FILE_NAME=requirements_cu129.txt"
set "DOWNLOAD_RUN=false"

(
    echo REQ_FILE_NAME=!REQ_FILE_NAME!
    echo DOWNLOAD_RUN=!DOWNLOAD_RUN!
) > "%CONFIG_FILE%"
echo Configuration saved.
echo.

:ConfigLoaded

:: Force requirements file to cu129
set "REQ_FILE_NAME=requirements_cu129.txt"
set "REQUIREMENTS=%APP_DIR%\%REQ_FILE_NAME%"
set "NEEDS_INSTALL=false"

if not exist "%PORTABLE_DIR%" mkdir "%PORTABLE_DIR%"

:: --- Step 1: Set up portable Git ---
if not exist "%GIT_DIR%\bin\git.exe" (
    echo Downloading PortableGit...
    powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%GIT_URL%', '%GIT_ZIP%'); exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to download PortableGit.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    echo Extracting PortableGit...
    mkdir "%GIT_DIR%" >nul 2>&1
    "%GIT_ZIP%" -y -o"%GIT_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to extract PortableGit.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    del "%GIT_ZIP%"
)

:: --- Step 2: Clone or update repository ---
if exist "%APP_DIR%" (
    if exist "%APP_DIR%\.git" (
        echo Repository exists. Checking for updates...
        pushd "%APP_DIR%"
        git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" checkout %BRANCH%
        git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" fetch

        for /f "tokens=*" %%i in ('git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" rev-parse HEAD') do set "LOCAL=%%i"
        for /f "tokens=*" %%i in ('git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" rev-parse origin/%BRANCH%') do set "REMOTE=%%i"

        if "!LOCAL!" neq "!REMOTE!" (
            echo Updates available on branch %BRANCH%.
            choice /c YN /m "Do you want to update? (This will discard local changes) (Y/N) "
            if !ERRORLEVEL! equ 1 (
                echo Resetting local repository to match remote...
                git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" reset --hard origin/%BRANCH%
                if !ERRORLEVEL! neq 0 (
                    echo ERROR: Failed to reset repository.
                    popd
                    set "PATH=%OLD_PATH%"
                    exit /b 1
                )
                echo Repository updated.
                set "NEEDS_INSTALL=true"
                set "DOWNLOAD_RUN=false"
                powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'DOWNLOAD_RUN=.*', 'DOWNLOAD_RUN=false' | Set-Content '%CONFIG_FILE%'"

                :: SELF-UPDATE CHECK
                popd
                call :self_update_check
            ) else (
                popd
            )
        ) else (
            echo Repository is up to date.
            popd
        )
    ) else (
        echo WARNING: %APP_DIR% exists but is not a git repo. Cleaning folder...
        rmdir /s /q "%APP_DIR%"
        echo Cloning repository on branch '%BRANCH%'...
        git clone --branch "%BRANCH%" "%REPO_URL%" "%APP_DIR%"
        if !ERRORLEVEL! neq 0 (
            echo ERROR: Failed to clone repository.
            set "PATH=%OLD_PATH%"
            exit /b 1
        )
        set "NEEDS_INSTALL=true"
        :: SELF-UPDATE CHECK after initial clone
        call :self_update_check
    )
) else (
    echo Cloning repository on branch '%BRANCH%'...
    git clone --branch "%BRANCH%" "%REPO_URL%" "%APP_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to clone repository.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    set "NEEDS_INSTALL=true"
    :: SELF-UPDATE CHECK after initial clone
    call :self_update_check
)

:: --- Step 3: Set up portable Python ---
if not exist "%PYTHON_DIR%\python.exe" (
    echo Checking Windows version...
    for /f "tokens=3 delims=." %%i in ('ver') do set WIN_BUILD=%%i

    if !WIN_BUILD! LSS 22000 (
        echo Windows 10 detected. Using portable full Python package.
        echo Downloading Python Nuget package...
        powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%PYTHON_NUGET_URL%', '%PYTHON_NUGET_ZIP%'); exit 0 } catch { exit 1 }"
        if !ERRORLEVEL! neq 0 (
            echo ERROR: Failed to download Python nuget package.
            set "PATH=%OLD_PATH%"
            exit /b 1
        )
        echo Extracting Python...
        set "TEMP_EXTRACT_DIR=%PORTABLE_DIR%\python_temp_extract"
        mkdir "!TEMP_EXTRACT_DIR!" >nul 2>&1
        powershell -Command "Expand-Archive -Path '%PYTHON_NUGET_ZIP%' -DestinationPath '!TEMP_EXTRACT_DIR!' -Force"

        :: The actual python files are in a 'tools' subdirectory inside the nuget package
        move "!TEMP_EXTRACT_DIR!\tools" "%PYTHON_DIR%"

        rmdir /s /q "!TEMP_EXTRACT_DIR!"
        del "%PYTHON_NUGET_ZIP%"

        echo Installing pip...
        powershell -Command "(New-Object Net.WebClient).DownloadFile('https://bootstrap.pypa.io/get-pip.py', '%PYTHON_DIR%\get-pip.py')"
        "%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
        del "%PYTHON_DIR%\get-pip.py"
    ) else (
        echo Windows 11 or newer detected. Using embeddable Python.
        echo Downloading Python Embeddable...
        powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%PYTHON_EMBED_URL%', '%PYTHON_ZIP%'); exit 0 } catch { exit 1 }"
        if !ERRORLEVEL! neq 0 (
            echo ERROR: Failed to download Python.
            set "PATH=%OLD_PATH%"
            exit /b 1
        )
        echo Extracting Python...
        mkdir "%PYTHON_DIR%" >nul 2>&1
        powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
        del "%PYTHON_ZIP%"

        set "PTH_FILE=%PYTHON_DIR%\python311._pth"
        if exist "!PTH_FILE!" (
            echo Enabling site packages in PTH file...
            powershell -Command "(Get-Content '!PTH_FILE!') -replace '#import site', 'import site' | Set-Content '!PTH_FILE!'"
        )

        echo Installing pip...
        powershell -Command "(New-Object Net.WebClient).DownloadFile('https://bootstrap.pypa.io/get-pip.py', '%PYTHON_DIR%\get-pip.py')"
        "%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location
        del "%PYTHON_DIR%\get-pip.py"
    )
)


:: --- Step 4: Set up uv ---
if not exist "%UV_DIR%\uv.exe" (
    echo Downloading uv...
    powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%UV_URL%', '%UV_ZIP%'); exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to download uv.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    echo Extracting uv...
    mkdir "%UV_DIR%" >nul 2>&1
    powershell -Command "Expand-Archive -Path '%UV_ZIP%' -DestinationPath '%UV_DIR%' -Force"
    del "%UV_ZIP%"
)

:: --- Step 5: Create virtual environment ---
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    "%UV_DIR%\uv.exe" venv "%VENV_DIR%" --python "%PYTHON_DIR%\python.exe"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to create virtual environment.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    set "NEEDS_INSTALL=true"
)

:: --- Step 6: Install dependencies (if needed) ---
if /I "!NEEDS_INSTALL!"=="true" (
    echo Installing/updating dependencies...
    if not exist "!REQUIREMENTS!" (
        echo ERROR: Requirements file not found: "!REQUIREMENTS!"
        echo Please check your configuration or the repository files.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    pushd "%APP_DIR%"
    "%UV_DIR%\uv.exe" pip install -r "!REQUIREMENTS!" --python "%VENV_DIR%\Scripts\python.exe"
    set "INSTALL_ERROR=!ERRORLEVEL!"
    popd
    if !INSTALL_ERROR! neq 0 (
        echo ERROR: Dependency installation failed.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    echo Dependencies installed successfully.

    echo Cleaning up package cache...
    "%UV_DIR%\uv.exe" cache clean

) else (
    echo Dependencies are up to date. Skipping installation.
)

:: --- Step 7: Run downloader (if needed) ---
if /I "!DOWNLOAD_RUN!"=="false" (
    if exist "%APP_DIR%\%DOWNLOAD_PY%" (
        echo Running download_models.py...
        pushd "%APP_DIR%"
        "%VENV_DIR%\Scripts\python.exe" "%DOWNLOAD_PY%"
        if !ERRORLEVEL! neq 0 (
            echo WARNING: download_models.py encountered an issue.
            echo Continuing anyway...
        ) else (
            powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'DOWNLOAD_RUN=.*', 'DOWNLOAD_RUN=true' | Set-Content '%CONFIG_FILE%'"
            set "DOWNLOAD_RUN=true"
        )
        popd
    ) else (
        echo WARNING: download_models.py not found. Skipping model download.
    )
) else (
    echo Model downloads already completed. Skipping...
)

:: --- Step 7.5: Set up FFmpeg ---
if not exist "%FFMPEG_PATH_VAR%\ffmpeg.exe" (
    echo Downloading FFmpeg...
    powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%FFMPEG_URL%', '%FFMPEG_ZIP%'); exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to download FFmpeg.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    echo Extracting FFmpeg...
    powershell -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%FFMPEG_EXTRACT_DIR%' -Force"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to extract FFmpeg.
        del "%FFMPEG_ZIP%"
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    del "%FFMPEG_ZIP%"
)

:: Restore original PATH
set "PATH=%OLD_PATH%"

:: --- Ensure LAUNCHER_ENABLED key exists after setup ---
:: Runs on first run or if the key is missing in existing config.

:: Detect if setup ran or key is missing
set "ADD_LAUNCHER_KEY=false"

if /I "!FIRST_RUN!"=="true" (
    set "ADD_LAUNCHER_KEY=true"
) else (
    if exist "%CONFIG_FILE%" (
        set "FOUND_KEY=false"
        for /f "usebackq tokens=1,* delims==" %%A in ("%CONFIG_FILE%") do (
            if /I "%%~A"=="LAUNCHER_ENABLED" set "FOUND_KEY=true"
        )
        if /I "!FOUND_KEY!"=="false" set "ADD_LAUNCHER_KEY=true"
    )
)

if /I "!ADD_LAUNCHER_KEY!"=="true" (
    echo.
    echo Ensuring LAUNCHER_ENABLED=1 is set in portable.cfg...
    if exist "%CONFIG_FILE%" (
        >>"%CONFIG_FILE%" echo LAUNCHER_ENABLED=1
    ) else (
        (echo LAUNCHER_ENABLED=1) > "%CONFIG_FILE%"
    )
)

:: --- Step 8: Run main application ---
if exist "%APP_DIR%\%MAIN_PY%" (
    echo.
    echo Starting main.py...
    echo ========================================
    pushd "%APP_DIR%"
    set "FFMPEG_PATH=%FFMPEG_PATH_VAR%"
    set "PATH=%FFMPEG_PATH_VAR%;%PATH%"
    "%VENV_DIR%\Scripts\python.exe" "%MAIN_PY%"
    popd
) else (
    echo ERROR: main.py not found in "%APP_DIR%".
    exit /b 1
)

echo.
echo Application closed. Press any key to exit...
pause >nul
endlocal
goto :eof

:self_update_check
set "ROOT_BAT=%BASE_DIR%Start_Portable.bat"
set "REPO_BAT=%APP_DIR%\Start_Portable.bat"
set "UPDATER_BAT=%PORTABLE_DIR%\update_start_portable.bat"

if not exist "%REPO_BAT%" (
    echo WARNING: Cannot find updated Start_Portable.bat in repository. Skipping self-update.
    goto :eof
)

echo Checking for launcher script updates...
fc /b "%ROOT_BAT%" "%REPO_BAT%" > nul
if errorlevel 1 (
    echo.
    echo ATTENTION: The launcher script Start_Portable.bat has been updated.
    echo The script will now restart itself to apply the changes.
    echo.

    (
        echo @echo off
        echo echo Waiting for main script to exit...
        echo timeout /t 3 /nobreak ^>nul
        echo echo Replacing Start_Portable.bat...
        echo copy /y "%REPO_BAT%" "%ROOT_BAT%"
        echo echo.
        echo echo Update complete. Relaunching...
        echo start "" /d "%BASE_DIR%" "Start_Portable.bat"
        echo exit
    ) > "%UPDATER_BAT%"

    start "" cmd /c "%UPDATER_BAT%"
    :: Use goto :eof to terminate the current script immediately after spawning the updater.
    :: This prevents the original script from continuing its execution.
    goto :eof
) else (
    echo Launcher script is up to date.
)
goto :eof
