set AGENT=%1
set ENVIRONMENT=%2


python train.py --environment %ENVIRONMENT% --model clac --agent %AGENT% --training_ts 1000000 --device-type cuda --random_training
python train.py --environment %ENVIRONMENT% --model sac  --agent %AGENT% --training_ts 1000000 --device-type cuda --random_training

python train.py --environment %ENVIRONMENT% --model clac --agent %AGENT% --training_ts 1000000 --device-type cuda 
python train.py --environment %ENVIRONMENT% --model sac  --agent %AGENT% --training_ts 1000000 --device-type cuda 