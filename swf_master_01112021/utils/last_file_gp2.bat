@echo off
FOR /F "delims=|" %%I IN ('DIR "C:\Users\abdel\Desktop\swf\swf_master_01112021\Sa-modat-mpo-pr\Data-Safi\" /B /O:D') do set NewestFile=%%I
echo %NewestFile%