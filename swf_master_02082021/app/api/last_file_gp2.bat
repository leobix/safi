@echo off
FOR /F "delims=|" %%I IN ('DIR "C:\Users\abdel\Desktop\swf_master\Sa-modat-mpo-pr\Data-Safi\" /B /O:D') do set NewestFile=%%I
echo %NewestFile%