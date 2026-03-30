@echo off
REM Run a Python script with MSVC x64 toolchain on PATH (for CUDA ext compilation).
REM Usage: run_with_msvc.bat <script.py> [args...]

setlocal

set VCVARS=""
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
)

if %VCVARS%=="" (
    echo [WARN] MSVC vcvarsall.bat not found — running without MSVC toolchain.
    echo        Kernel compilation will fail; pre-built .pyd will be used if present.
) else (
    call %VCVARS% x64
)

cd /d "%~dp0..\.."
.venv\Scripts\python %*
endlocal
