@echo off
setlocal EnableExtensions
REM Teacher play launcher (cmd.exe). Use the SAME isaaclab.bat + conda env that work for train_teacher_policy.py.
REM If Kit fails with omni.physx.stageupdate / omni.kit.usd, fix Isaac Lab <-> Isaac Sim version pairing (installer issue).

REM HOVER repo root (this file is in scripts\rsl_rl\)
set "HOVER_ROOT=%~dp0..\.."
for %%I in ("%HOVER_ROOT%") do set "HOVER_ROOT=%%~fI"

REM === Edit if your Isaac Lab install differs ===
if not defined ISAACLAB_BAT set "ISAACLAB_BAT=C:\Users\kylel\IsaacLab\isaaclab.bat"

REM === Match your run folder and checkpoint file ===
set "RUN_DIR=%HOVER_ROOT%\logs\teacher\26_04_12_17-51-07"
set "CHECKPOINT=model_1000.pt"

if not exist "%ISAACLAB_BAT%" (
  echo ERROR: ISAACLAB_BAT not found: "%ISAACLAB_BAT%"
  echo Set ISAACLAB_BAT to the isaaclab.bat that launches training, then re-run.
  exit /b 1
)

REM reference_motion_path is optional: play.py reads motion_path from RUN_DIR\env_config.json when omitted.
REM Default: headless + MP4s + TRAIN-like random motions on reset (like teacher training). Add --no_randomize for deterministic TEST mode + flat terrain.
REM Add --debug_spawn to print env0 root quat / projected gravity vs reference (spot upside-down spawn vs policy collapse).
REM GUI Kit often fails on py3.11+Sim4.5 — add --gui only if it starts on your machine.
"%ISAACLAB_BAT%" -p "%HOVER_ROOT%\scripts\rsl_rl\play.py" ^
  --robot h1 ^
  --num_envs 8 ^
  --teacher_policy.resume_path "%RUN_DIR%" ^
  --teacher_policy.checkpoint "%CHECKPOINT%"

endlocal
exit /b %ERRORLEVEL%
