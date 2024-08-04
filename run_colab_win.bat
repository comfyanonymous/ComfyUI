echo Make sure you set up your branch name in colab_runner.ipynb

IF NOT EXIST ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install pydrive
python connect_to_colab.py
deactivate
