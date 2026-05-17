@echo off
setlocal EnableExtensions
REM Preview a motion clip in Isaac Sim (no training). Edit HOVER_ROOT / ISAACLAB_BAT / CLIP_INDEX as needed.

set "HOVER_ROOT=%~dp0..\.."
for %%I in ("%HOVER_ROOT%") do set "HOVER_ROOT=%%~fI"

if not defined ISAACLAB_BAT set "ISAACLAB_BAT=C:\Users\kylel\IsaacLab\isaaclab.bat"

set "MOTION_PKL=%HOVER_ROOT%\neural_wbc\data\data\motions\amass_all.pkl"
set "CLIP_INDEX=0"

if not exist "%ISAACLAB_BAT%" (
  echo ERROR: ISAACLAB_BAT not found: "%ISAACLAB_BAT%"
  exit /b 1
)

REM List clips without Kit:  python scripts\rsl_rl\view_reference_motion.py --list_clips_only --reference_motion_path "%MOTION_PKL%"
REM Headless MP4 (same Kit as training play): add --record_video  (optional: --video_dir "%HOVER_ROOT%\logs\reference_motion_preview")

"%ISAACLAB_BAT%" -p "%HOVER_ROOT%\scripts\rsl_rl\view_reference_motion.py" ^
  --robot h1 ^
  --reference_motion_path "%MOTION_PKL%" ^
  --clip_index %CLIP_INDEX%

endlocal
exit /b %ERRORLEVEL%
