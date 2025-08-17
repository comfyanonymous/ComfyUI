#!/usr/bin/env python3

import subprocess
import os
import re
import datetime
import time
import argparse

from __build__ import build, log_step, log_step_info

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FILE_PY_PROJECT = os.path.join(_THIS_DIR, 'pyproject.toml')

parser = argparse.ArgumentParser()
parser.add_argument(
  "-m", "--message", help="The git commit message", required=True, action="store", type=str
)
args = parser.parse_args()

start = time.time()
build()

log_step(msg='Updating version in pyproject.toml')
py_project = ''
with open(_FILE_PY_PROJECT, "r", encoding='utf-8') as f:
  py_project = f.read()

version = re.search(r'^\s*version\s*=\s*"(.*?)"', py_project, flags=re.MULTILINE)
version_old = version[1]

now = datetime.datetime.now()
version_new = version_old.split('.')
version_new[-1] = f'{str(now.year)[2:]}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}'
version_new = '.'.join(version_new)

log_step_info(f'Updating from v{version_old} to v{version_new}')
py_project = py_project.replace(version_old, version_new)
with open(_FILE_PY_PROJECT, "w", encoding='utf-8') as f:
  f.write(py_project)
log_step(status="Done")

log_step('Running git add')
process = subprocess.Popen(['git', 'add', '.'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
log_step(status="Done")

log_step('Running git commit')
process = subprocess.Popen(['git', 'commit', '-a', '-v', '-m', args.message],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
log_step(status="Done")

print(f'Finished all in {round(time.time() - start, 3)}s')
