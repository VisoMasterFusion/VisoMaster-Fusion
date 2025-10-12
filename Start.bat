@echo off
REM Check if .venv directory exists
IF EXIST ".venv" (
    echo Found .venv, activating virtual uv based environment...
    call ".venv\Scripts\activate"
) ELSE (
    echo .venv not found, activating conda environment "visomaster"...
    call conda activate visomaster
)

REM Run main.py
echo Running VisoMaster...
python main.py
