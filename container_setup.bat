@echo off
setlocal enabledelayedexpansion
REM Run this batch in powershell
REM Find IP Address
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set "ip=%%i"
)

REM Removinig trailing space from the ip
set "ip=%ip:~1%"
echo Current IP: %ip%

REM Set DISPLAY environment variable
set "DISPLAY=!ip!:0.0"
echo Exporting display to %DISPLAY%

REM Run the docker container
docker run -it --rm --gpus all -v .:/pyperbot_v2 -e DISPLAY=%DISPLAY% pyperbotv2:v2
echo Container setup completed
endlocal