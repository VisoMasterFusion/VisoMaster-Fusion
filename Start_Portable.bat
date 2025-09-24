@echo off
setlocal EnableDelayedExpansion

:: --- Basic Setup ---
:: Define repo details
set "REPO_URL=https://github.com/Glat0s/VisoMaster.git"
set "BRANCH=fusion"

:: Extract repo name from URL
for %%a in ("%REPO_URL%") do set "REPO_NAME=%%~na"

:: Define paths
set "BASE_DIR=%~dp0"
set "PORTABLE_DIR=%BASE_DIR%portable-files"
set "APP_DIR=%BASE_DIR%%REPO_NAME%"
set "PYTHON_DIR=%PORTABLE_DIR%\python"
set "UV_DIR=%PORTABLE_DIR%\uv"
set "GIT_DIR=%PORTABLE_DIR%\git"
set "VENV_DIR=%PORTABLE_DIR%\venv"
set "PYTHON_EMBED_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
set "PYTHON_ZIP=%PORTABLE_DIR%\python-embed.zip"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP=%PORTABLE_DIR%\uv.zip"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.51.0.windows.1/PortableGit-2.51.0-64-bit.7z.exe"
set "GIT_ZIP=%PORTABLE_DIR%\PortableGit.exe"
set "CONFIG_FILE=%BASE_DIR%portable.cfg"
set "DOWNLOAD_PY=%APP_DIR%\download_models.py"
set "MAIN_PY=%APP_DIR%\main.py"

:: --- Step 0: User Configuration ---
:: Read config or prompt user for the first time
if exist "%CONFIG_FILE%" (
    echo Loading configuration from portable.cfg...
    call "%CONFIG_FILE%"
) else (
    echo.
    echo Welcome! Please select the environment to install.
    echo This choice will be saved in portable.cfg for future runs.
    echo --------------------------------------------------------
    echo 1. Install for CUDA 11.8 (requirements_cu118.txt)
    echo 2. Install for CUDA 12.4 (requirements_cu124.txt)
    echo 3. Install for CUDA 12.8 (requirements_cu128.txt)
    echo 4. Install for RTX 50xx cards (requirements_rtx50.txt)
    choice /c 1234 /m "Enter your choice (1-4): "
    
    set "CHOICE=!ERRORLEVEL!"
    if !CHOICE! equ 1 set "REQ_FILE_NAME=requirements_cu118.txt"
    if !CHOICE! equ 2 set "REQ_FILE_NAME=requirements_cu124.txt"
    if !CHOICE! equ 3 set "REQ_FILE_NAME=requirements_cu128.txt"
    if !CHOICE! equ 4 set "REQ_FILE_NAME=requirements_rtx50.txt"

    set "REQUIREMENTS=%APP_DIR%\!REQ_FILE_NAME!"
    set "DOWNLOAD_RUN=false"

    :: Write to config file in a safe format
    (
        echo @echo off
        echo set "REQUIREMENTS=!REQUIREMENTS!"
        echo set "DOWNLOAD_RUN=false"
    )>"%CONFIG_FILE%"
    echo Configuration saved.
    echo.
)

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
        exit /b 1
    )

    echo Extracting PortableGit...
    mkdir "%GIT_DIR%" >nul 2>&1
    "%GIT_ZIP%" -y -o"%GIT_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to extract PortableGit.
        exit /b 1
    )
    del "%GIT_ZIP%"
)

:: Set Git executable path
set "GIT=%GIT_DIR%\bin\git.exe"

:: --- Step 2: Clone or update repository ---
if not exist "%APP_DIR%\.git" (
    echo Cloning repository on branch '%BRANCH%'...
    "%GIT%" clone --branch "%BRANCH%" "%REPO_URL%" "%APP_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to clone repository.
        exit /b 1
    )
) else (
    echo Checking for updates...
    pushd "%APP_DIR%"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to access repository directory "%APP_DIR%".
        exit /b 1
    )
    
    "%GIT%" checkout "%BRANCH%" >nul 2>&1
    "%GIT%" fetch
    for /f %%i in ('"%GIT%" rev-parse HEAD') do set "LOCAL=%%i"
    for /f %%i in ('"%GIT%" rev-parse "origin/%BRANCH%"') do set "REMOTE=%%i"
    
    if "!LOCAL!" neq "!REMOTE!" (
        echo Updates available for the repository on branch %BRANCH%.
        choice /c YN /m "Do you want to update? (Y/N) "
        if !ERRORLEVEL! equ 1 (
            "%GIT%" pull
            if !ERRORLEVEL! neq 0 (
                echo ERROR: Failed to pull updates.
                popd
                exit /b 1
            )
            echo Repository updated.
            set "NEEDS_INSTALL=true"
        )
    ) else (
        echo Repository is up to date.
    )
    popd
)

:: Verify requirements file exists after clone/update
if not exist "!REQUIREMENTS!" (
    echo ERROR: Selected requirements file "!REQUIREMENTS!" not found.
    echo Please check your repository or delete portable.cfg to choose again.
    exit /b 1
)

:: --- Step 3: Set up portable Python ---
if not exist "%PYTHON_DIR%\python.exe" (
    echo Downloading Python Embeddable...
    powershell -Command "try { (New-Object Net.WebClient).DownloadFile('%PYTHON_EMBED_URL%', '%PYTHON_ZIP%'); exit 0 } catch { exit 1 }"
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to download Python.
        exit /b 1
    )

    echo Extracting Python...
    mkdir "%PYTHON_DIR%" >nul 2>&1
    powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
    del "%PYTHON_ZIP%"

    set "PTH_FILE=%PYTHON_DIR%\python313._pth"
    if exist "%PTH_FILE%" (
        echo Enabling pip...
        powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site', 'import site' | Set-Content '%PTH_FILE%'"
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
        exit /b 1
    )
    set "NEEDS_INSTALL=true"
)

:: --- Step 6: Install dependencies (if needed) ---
if /I "!NEEDS_INSTALL!"=="true" (
    echo Installing/updating dependencies using uv and pyproject.toml...

    :: Install using uv. It will automatically find pyproject.toml in the app directory.
    pushd "%APP_DIR%"
    "%UV_DIR%\uv.exe" pip install --python "%VENV_DIR%\Scripts\python.exe" -r "!REQUIREMENTS!"
    set "INSTALL_ERROR=!ERRORLEVEL!"
    popd

    if !INSTALL_ERROR! neq 0 (
        echo ERROR: Dependency installation failed. Check pyproject.toml and requirements files.
        exit /b 1
    )
    echo Dependencies installed successfully.
) else (
    echo Dependencies are assumed to be up to date.
)

:: --- Step 7: Run downloader ---
if /I "!DOWNLOAD_RUN!"=="false" (
    if exist "%DOWNLOAD_PY%" (
        echo Running download_models.py...
        "%VENV_DIR%\Scripts\python.exe" "%DOWNLOAD_PY%"
        if !ERRORLEVEL! neq 0 (
            echo ERROR: download_models.py failed.
            exit /b 1
        )
        powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'set \"DOWNLOAD_RUN=false\"', 'set \"DOWNLOAD_RUN=true\"' | Set-Content '%CONFIG_FILE%'"
    ) else (
        echo WARNING: download_models.py not found. Skipping.
        powershell -Command "(Get-Content '%CONFIG_FILE%') -replace 'set \"DOWNLOAD_RUN=false\"', 'set \"DOWNLOAD_RUN=true\"' | Set-Content '%CONFIG_FILE%'"
    )
)

:: --- Step 8: Run main application ---
if exist "%MAIN_PY%" (
    echo Starting main.py...
    "%VENV_DIR%\Scripts\python.exe" "%MAIN_PY%"
) else (
    echo ERROR: main.py not found in "%APP_DIR%".
    exit /b 1
)

echo Done.
endlocal