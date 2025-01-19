#!/bin/bash

cd /home/pi01/cat-detector/

source ./.venv/bin/activate

python ./src/detector.py --resolution 640x480

