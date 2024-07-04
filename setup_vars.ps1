# Remove duplicates of $_PATH
$env:PATH = ($env:PATH -split ';' | Select-Object -Unique) -join ';'

Invoke-BatchFile 'C:\Program Files (x86)\Intel\oneAPI\setvars.bat'