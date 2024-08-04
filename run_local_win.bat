echo Make sure you set up your branch name in colab_runner.ipynb

IF NOT EXIST ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py --dont-print-server
call .venv\Scripts\deactivate

echo Script execution completed.
pause
