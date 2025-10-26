@echo off
setlocal EnableDelayedExpansion

:: --- Basic Setup ---
:: Define repo details
set "REPO_URL=https://github.com/VisoMasterFusion/VisoMaster-Fusion.git"
set "BRANCH=main"

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
set "PYTHON_EMBED_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PYTHON_ZIP=%PORTABLE_DIR%\python-embed.zip"
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
:: Read config or prompt user for the first time
if exist "%CONFIG_FILE%" (
    echo Loading configuration from portable.cfg...
    for /f "usebackq tokens=1,* delims==" %%a in ("%CONFIG_FILE%") do set "%%a=%%b"
    goto :ConfigLoaded
)

:: First time setup
set "REQ_FILE_NAME=requirements_cu129.txt"
set "DOWNLOAD_RUN=false"

:: Write to config file in a clean "KEY=VALUE" format.
(
    echo REQ_FILE_NAME=!REQ_FILE_NAME!
    echo DOWNLOAD_RUN=!DOWNLOAD_RUN!
) > "%CONFIG_FILE%"
echo Configuration saved.
echo.

:ConfigLoaded

:: Force requirements file to cu129
set "REQ_FILE_NAME=requirements_cu129.txt"

:: Reconstruct the full path to the requirements file from the config.
set "REQUIREMENTS=%APP_DIR%\%REQ_FILE_NAME%"

:: This flag will determine if we need to run pip install.
set "NEEDS_INSTALL=false"

:: Create the directory for portable tools if it doesn't exist
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
        :: Folder exists and is a git repo, check for updates
        pushd "%APP_DIR%"
        git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" checkout %BRANCH%
        git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" fetch

        for /f "tokens=*" %%i in ('git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" rev-parse HEAD') do set "LOCAL=%%i"
        for /f "tokens=*" %%i in ('git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" rev-parse origin/%BRANCH%') do set "REMOTE=%%i"

        if "!LOCAL!" neq "!REMOTE!" (
            echo Updates available on branch %BRANCH%.
            choice /c YN /m "Do you want to update? (Y/N) "
            if !ERRORLEVEL! equ 1 (
                git --git-dir="%APP_DIR%\.git" --work-tree="%APP_DIR%" pull
                if !ERRORLEVEL! neq 0 (
                    echo ERROR: Failed to pull updates.
                    popd
                    set "PATH=%OLD_PATH%"
                    exit /b 1
                )
                echo Repository updated.
                set "NEEDS_INSTALL=true"
                :: Reset download flag when updating
                set "DOWNLOAD_RUN=false"
                powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'DOWNLOAD_RUN=.*', 'DOWNLOAD_RUN=false' | Set-Content '%CONFIG_FILE%'"
            )
        ) else (
            echo Repository is up to date.
        )
        popd
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
)

:: --- Step 3: Set up portable Python ---
if not exist "%PYTHON_DIR%\python.exe" (
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

    :: Enable site packages
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
        echo ERROR: Dependency installation failed. Check your requirements file.
        set "PATH=%OLD_PATH%"
        exit /b 1
    )
    echo Dependencies installed successfully.
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
            :: Update config file only on success
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
