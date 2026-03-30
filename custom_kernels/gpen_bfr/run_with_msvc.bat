@echo off
REM Run a Python script with MSVC environment for CUDA kernel compilation.
REM Usage: run_with_msvc.bat custom_kernels/gpen_bfr/benchmark_gpen.py [args...]

setlocal

REM Locate MSVC vcvars64.bat
set VCVARS=
for /d %%I in ("C:\Program Files\Microsoft Visual Studio\2022\*") do (
    if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" set VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat
)
for /d %%I in ("C:\Program Files (x86)\Microsoft Visual Studio\2019\*") do (
    if not defined VCVARS (
        if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" set VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat
    )
)

if not defined VCVARS (
    echo [run_with_msvc] MSVC not found - running without it (PyTorch fallback will be used)
    goto run
)
echo [run_with_msvc] Using MSVC: %VCVARS%
call "%VCVARS%" >nul 2>&1

:run
cd /d "%~dp0..\.."
.venv\Scripts\python %*
