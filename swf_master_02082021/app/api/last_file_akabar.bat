@echo off
FOR /F "delims=|" %%I IN ('DIR "C:\Users\abdel\Desktop\swf_master\Sa-modat-mpo-pr\AKABAR\" /B /O:D') do set NewestFile=%%I
echo %NewestFile%