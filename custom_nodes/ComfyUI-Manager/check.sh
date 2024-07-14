#!/bin/bash

echo
echo CHECK1

files=(
    "custom-node-list.json"
    "model-list.json"
    "alter-list.json"
    "extension-node-map.json"
    "github-stats.json"
    "node_db/new/custom-node-list.json"
    "node_db/new/model-list.json"
    "node_db/new/extension-node-map.json"
    "node_db/dev/custom-node-list.json"
    "node_db/dev/model-list.json"
    "node_db/dev/extension-node-map.json"
    "node_db/tutorial/custom-node-list.json"
    "node_db/tutorial/model-list.json"
    "node_db/tutorial/extension-node-map.json"
    "node_db/legacy/custom-node-list.json"
    "node_db/legacy/model-list.json"
    "node_db/legacy/extension-node-map.json"
    "node_db/forked/custom-node-list.json"
    "node_db/forked/model-list.json"
    "node_db/forked/extension-node-map.json"
)

for file in "${files[@]}"; do
    python json-checker.py "$file"
done

echo
echo CHECK2
find ~/.tmp/default -name "*.py" -print0 | xargs -0 grep "crypto"

echo
echo CHECK3
find ~/.tmp/default -name "requirements.txt" | xargs grep "^\s*https\\?:"
find ~/.tmp/default -name "requirements.txt" | xargs grep "\.whl"

echo
