@echo off
FOR /F "delims=|" %%I IN ('DIR "\\Sa-modat-mto-pr\Data-Safi\" /B /O:D') do set NewestFile=%%I
echo %NewestFile%