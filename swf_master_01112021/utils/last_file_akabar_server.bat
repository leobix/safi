@echo off
FOR /F "delims=|" %%I IN ('DIR "\\Sa-modat-mto-pr\akabar\" /B /O:D') do set NewestFile=%%I
echo %NewestFile%