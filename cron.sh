#!/bin/bash
#make-run.sh
#make sure a process is always running.

export DISPLAY=:0 #needed if you are running a simple gui app.

process="src/detector.py"
makerun="/home/pi01/cat-detector/start.sh"

if pgrep -f ${process} > /dev/null 2>&1 
then
    echo "Already running..."
    exit
else
    echo "Process not found, running..."
    $makerun &
fi

exit

