@echo off
call "D:\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d "D:\VisoMaster - fusion\VisoMaster-fusion-git-dev"
.venv\Scripts\python %*
