@echo off
REM Aller dans le dossier du projet
cd /d "C:\Users\benja\Documents\GitHub\ComfyUI triton sageAttention\ComfyUI"

REM Activer l'environnement virtuel
call venv\Scripts\activate.bat

REM Afficher un message de confirmation
echo [OK] Environnement virtuel activé.

REM Lancer ton script Python (change "main.py" si besoin)
python main.py