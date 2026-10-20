@echo off
rem Start Gufo with Qwen3.8-Flash-Next UD-Q4_K_XL (96 GB carve, port 8888).
rem Matches the historical launch: latin draft vocab, survival policy, prompt lookup.
rem Loader events stream to console AND gufo-server-log.txt (as before).
setlocal
set "ARGS=%*"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-gufo-flash-next-xl.ps1" -Lookup %ARGS%
