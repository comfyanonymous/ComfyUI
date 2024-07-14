#!/bin/bash
rm ~/.tmp/dev/*.py > /dev/null 2>&1
python ../../scanner.py ~/.tmp/dev
