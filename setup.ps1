$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r requirements.txt
Write-Host "Accept the LTX-2.5 HF license, run hf auth login for the model download, then .venv\Scripts\python.exe run.py."