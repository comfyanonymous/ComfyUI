import shutil
from google.colab import drive, notebooks

drive.mount('/content/drive')

local_file_path = "./colab_runner.ipynb"
drive_file_path = '/content/drive/MyDrive/colab_runner.ipynb'
shutil.copy(local_file_path, drive_file_path)

notebooks.run_notebook(drive_file_path)