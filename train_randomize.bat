set ENVIRONMENT=%1

python train.py --environment %ENVIRONMENT% --model clac --agent a1 --training_ts 1000000 --device_type cuda 
python train.py --environment %ENVIRONMENT% --model sac  --agent a1 --training_ts 1000000 --device_type cuda 

python train.py --environment %ENVIRONMENT% --model clac --agent a2 --training_ts 1000000 --device_type cuda 
python train.py --environment %ENVIRONMENT% --model sac  --agent a2 --training_ts 1000000 --device_type cuda 

python train.py --environment %ENVIRONMENT% --model clac --agent a3 --training_ts 1000000 --device_type cuda 
python train.py --environment %ENVIRONMENT% --model sac  --agent a3 --training_ts 1000000 --device_type cuda 


rem  training time: 1M for all 
rem  ./train.bat AntPyBulletEnv-v0
rem  ./train.bat HalfCheetahPyBulletEnv-v0
rem  ./train.bat HopperPyBulletEnv-v0
rem  ./train.bat Walker2DPyBulletEnv-v0
rem  ./train.bat ReacherPyBulletEnv-v0
rem  ./train.bat InvertedDoublePendulumPyBulletEnv-v0