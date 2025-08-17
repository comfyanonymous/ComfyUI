import os

from .utils import load_json_file, path_exists, save_json_file

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
USERDATA = os.path.join(THIS_DIR, '..', 'userdata')


def read_userdata_file(rel_path: str):
  """Reads a file from the userdata directory."""
  file_path = clean_path(rel_path)
  if path_exists(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
      return file.read()
  return None


def save_userdata_file(rel_path: str, content: str):
  """Saves a file from the userdata directory."""
  file_path = clean_path(rel_path)
  with open(file_path, 'w+', encoding='UTF-8') as file:
    file.write(content)


def delete_userdata_file(rel_path: str):
  """Deletes a file from the userdata directory."""
  file_path = clean_path(rel_path)
  if os.path.isfile(file_path):
    os.remove(file_path)


def read_userdata_json(rel_path: str):
  """Reads a json file from the userdata directory."""
  file_path = clean_path(rel_path)
  return load_json_file(file_path)


def save_userdata_json(rel_path: str, data: dict):
  """Saves a json file from the userdata directory."""
  file_path = clean_path(rel_path)
  return save_json_file(file_path, data)


def clean_path(rel_path: str):
  """Cleans a relative path by splitting on forward slash and os.path.joining."""
  cleaned = USERDATA
  paths = rel_path.split('/')
  for path in paths:
    cleaned = os.path.join(cleaned, path)
  return cleaned
