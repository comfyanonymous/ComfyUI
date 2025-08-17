#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

{
  black --version | grep -E "23\." > /dev/null
} || {
  echo "Linter requires 'black==23.*' !"
  exit 1
}

ISORT_VERSION=$(isort --version-number)
if [[ "$ISORT_VERSION" != 5.12* ]]; then
  echo "Linter requires isort==5.12.0 !"
  exit 1
fi

echo "Running isort ..."
isort . --atomic

echo "Running black ..."
black -l 100 .

echo "Running flake8 ..."
if [ -x "$(command -v flake8)" ]; then
  flake8 .
else
  python3 -m flake8 .
fi

echo "Running mypy..."

mypy --exclude 'setup.py|notebooks' .
