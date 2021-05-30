echo off
set AGENT=%1


python train.py --environment HopperPyBulletEnv-v0 --model sac --agent %AGENT% --training_ts 1000000 --device-type cuda 
python train.py --environment HopperPyBulletEnv-v0 --model clac --agent %AGENT% --training_ts 1000000 --device-type cuda 

python train.py --environment AntPyBulletEnv-v0 --model sac --agent %AGENT% --training_ts 3000000 --device-type cuda 
python train.py --environment AntPyBulletEnv-v0 --model clac --agent %AGENT% --training_ts 3000000 --device-type cuda 

python train.py --environment ReacherPyBulletEnv-v0 --model sac --agent %AGENT% --training_ts 10000000 --device-type cuda 
python train.py --environment ReacherPyBulletEnv-v0 --model clac --agent %AGENT% --training_ts 10000000 --device-type cuda 

python train.py --environment HumanoidPyBulletEnv-v0 --model sac --agent %AGENT% --training_ts 10000000 --device-type cuda 
python train.py --environment HumanoidPyBulletEnv-v0 --model clac --agent %AGENT% --training_ts 10000000 --device-type cuda 

python train.py --environment HumanoidFlagrunPyBulletEnv-v0 --model sac --agent %AGENT% --training_ts 10000000 --device-type cuda 
python train.py --environment HumanoidFlagrunPyBulletEnv-v0 --model clac --agent %AGENT% --training_ts 10000000 --device-type cuda 

python train.py --environment HumanoidFlagrunHarderPyBulletEnv-v0 --model sac --agent %AGENT% --training_ts 10000000 --device-type cuda 
python train.py --environment HumanoidFlagrunHarderPyBulletEnv-v0 --model clac --agent %AGENT% --training_ts 10000000 --device-type cuda 