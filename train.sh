#!/bin/bash


python train.py --environment "$2" --model clac --agent "$1" --training_ts 3000000 --device_type cuda --random_training --random_testing
python train.py --environment "$2"--model sac  --agent "$1" --training_ts 3000000 --device_type cuda --random_training --random_testing

python train.py --environment "$2" --model clac --agent "$1" --training_ts 3000000 --device_type cuda --random_testing
python train.py --environment "$2" --model sac  --agent "$1" --training_ts 3000000 --device_type cuda --random_testing