#!/bin/bash
CUSTOM_NODE_DIR="/comfy-ts/custom_nodes"
VENV_DIR="/venv"
. "$VENV_DIR"/bin/activate

# Ensure the custom_nodes directory exists
mkdir -p "$CUSTOM_NODE_DIR"

cd "$CUSTOM_NODE_DIR"
counter=0
errors=0

while read URL BRANCH COMMIT
    do
    cd "$CUSTOM_NODE_DIR"
    echo Pulling repo "$URL" into `pwd`
    git clone "$URL"
    if [ $? -ne 0 ]
    then
        let errors=$errors+1
    fi
    DIR=$(basename "$URL")
    cd "$DIR"
    git switch "$BRANCH"
    git reset --hard "$COMMIT"
    pip install -r requirements.txt
   let  counter=$counter+1
done < /usr/share/custom_nodes/repos.txt

#Nailing down opencv version:

pip uninstall -y opencv-contrib-python opencv-python opencv-python-headless
pip install opencv-python==4.7.0.72

rm -rf ~/.cache/pip/*

echo $counter custom nodes pulled\; $errors errors encountered.
