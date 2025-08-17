#!/usr/bin/env python3

# A nicer output for git pulling custom nodes (and ComfyUI).
# Quick shell version: ls | xargs -I % sh -c 'echo; echo %; git -C % pull'

import os
from subprocess import Popen, PIPE, STDOUT


def pull_path(path):
  p = Popen(["git", "-C", path, "pull"], stdout=PIPE, stderr=STDOUT)
  output, error = p.communicate()
  return output.decode()

THIS_DIR=os.path.dirname(os.path.abspath(__file__))

def show_output(output):
  if output.startswith('Already up to date'):
    print(f' \33[32m🗸 {output}\33[0m', end ='')
  elif output.startswith('error:'):
    print(f' \33[31m🞫 Error.\33[0m \n {output}')
  else:
    print(f' \33[33m🡅 Needs update.\33[0m \n {output}', end='')


os.chdir(THIS_DIR)
os.chdir("../")

# Get the list or custom nodes, so we can format the output a little more nicely.
custom_extensions = []
custom_extensions_name_max = 0
for directory in os.listdir(os.getcwd()):
  if os.path.isdir(directory) and directory != "__pycache__": #and directory != "rgthree-comfy" :
    custom_extensions.append({
      'directory': directory
    })
    if len(directory) > custom_extensions_name_max:
      custom_extensions_name_max = len(directory)

if len(custom_extensions) == 0:
  custom_extensions_name_max = 15
else:
  custom_extensions_name_max += 6

# Update ComfyUI itself.
label = "{0:.<{max}}".format('Updating ComfyUI ', max=custom_extensions_name_max)
print(label, end = '')
show_output(pull_path('../'))

# If we have custom nodes, update them as well.
if len(custom_extensions) > 0:
  print(f'\nUpdating custom_nodes ({len(custom_extensions)}):')
  for custom_extension in custom_extensions:
    directory = custom_extension['directory']
    label = "{0:.<{max}}".format(f'🗀  {directory} ', max=custom_extensions_name_max)
    print(label, end = '')
    show_output(pull_path(directory))
