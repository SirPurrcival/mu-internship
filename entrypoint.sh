#!/bin/bash

## Add nest variables to path
source /opt/nest/bin/nest_vars.sh

## Change WD
cd code/simulation

## Run the script
python3 run.py

while true
do
	echo "The simulation is done, you can now shutdown the pod"
	sleep 100
done
