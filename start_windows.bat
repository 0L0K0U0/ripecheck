@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title RipeCheck launcher
set "ENV_NAME=ripecheck"

rem If Anaconda is installed in an unusual folder, remove "rem" from the next line and fix the path.
set "RW_CONDA=D:\tools\anaconda"

rem ---- 1. Find Anaconda or Miniconda ----
if not defined RW_CONDA (
  for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "C:\anaconda3" "C:\miniconda3" "D:\anaconda3" "D:\miniconda3") do (
    if not defined RW_CONDA if exist "%%~D\Scripts\activate.bat" set "RW_CONDA=%%~D"
  )
)
if not defined RW_CONDA (
  for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
    if not defined RW_CONDA for %%R in ("%%~dpI..") do if exist "%%~fR\Scripts\activate.bat" set "RW_CONDA=%%~fR"
  )
)
if not defined RW_CONDA (
  echo Could not find Anaconda or Miniconda.
  echo Open this file in Notepad and set RW_CONDA near the top.
  goto :fail
)
echo Using conda at %RW_CONDA%
call "%RW_CONDA%\Scripts\activate.bat" "%RW_CONDA%"
if errorlevel 1 goto :fail

rem ---- 2. Create the environment on first run ----
call conda activate %ENV_NAME% >nul 2>nul
python -c "import sys; sys.exit(0 if '%ENV_NAME%' in sys.prefix else 1)" >nul 2>nul
if errorlevel 1 (
  echo First run: creating the %ENV_NAME% environment. This takes a few minutes.
  call conda create -n %ENV_NAME% python=3.11 -y
  if errorlevel 1 goto :fail
  call conda activate %ENV_NAME%
  if errorlevel 1 goto :fail
)

rem ---- 3. Install packages if any are missing ----
python -c "import flask, cv2, numpy" >nul 2>nul
if errorlevel 1 (
  echo Installing packages. This only happens once.
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
)

rem ---- 4. Run. The app opens your browser by itself. ----
echo.
echo Starting RipeCheck at http://127.0.0.1:5057
echo To stop it, press Ctrl+C in this window, or just close the window.
echo.
python app.py
echo.
echo RipeCheck has stopped.
pause
goto :end

:fail
echo.
echo Something went wrong. Copy the text in this window and send it to whoever is helping you.
pause
exit /b 1

:end
endlocal
