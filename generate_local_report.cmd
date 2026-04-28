@echo off
setlocal
set UNIVERSE=%1
if "%UNIVERSE%"=="" set UNIVERSE=india-largecap
python india_swing_scanner.py %UNIVERSE%
endlocal
