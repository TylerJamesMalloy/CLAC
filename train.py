import gym
import pybulletgym
import numpy as np
import pandas as pd 
import argparse
import time

from torch._C import device

from clac import CLAC
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure


# parse input arguments 

# model type 
# learning method 

def evaluate(model, env, arglist):
    episode_rewards = []
    obs = env.reset()
    episode_reward = 0
    doneEvaluating = False
    while not doneEvaluating:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward 
        if done:
            episode_rewards.append(episode_reward)
            obs = env.reset()
            episode_reward = 0

            if arglist.random_testing: 
                env.unwrapped.randomize()
        
        if(len(episode_rewards) >= arglist.evaluate_es):
            doneEvaluating = True  
    
    if arglist.random_testing: 
        env.unwrapped.reset_features() 
    return np.mean(episode_rewards)

def train(args):
    env = gym.make(arglist.environment)
    
    training_type = 'randomized' if arglist.random_training else 'normal' 
    save_folder = "./trained_models/" + arglist.environment + "/" + arglist.model + "/" + training_type + "/" + arglist.agent + "/" # each agent gets its own folder 

    if(arglist.model == "clac"):
        model = CLAC("MlpPolicy", env, verbose=1, target_mutual_information="auto", device=arglist.device_type, arglist=arglist)
    elif(arglist.model == "sac"):
        model = SAC("MlpPolicy", env, verbose=1, target_entropy="auto", device=arglist.device_type, arglist=arglist)
    
    new_logger = configure(save_folder, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps= arglist.training_ts, log_interval= 100)
    model.save(save_folder + "/" + arglist.agent)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--environment", type=str, default="Pendulum-v0", help="training environment to train on")
    parser.add_argument("--model", type=str, default="clac", help="model type to train on")
    parser.add_argument("--agent", type=str, default="a1", help="name of agent, for saving results")
    parser.add_argument("--training_ts", type=int, default=1000000, help="number of time steps to train on")
    parser.add_argument("--training_evals", type=int, default=1, help="number of time steps to train on")
    parser.add_argument("--random_training", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--evaluate_es", type=int, default=100, help="number of time steps to evaluate trained model")
    parser.add_argument("--random_eval", action="store_true", default=False)
    parser.add_argument("--device_type", type=str, default="cuda", help="one of cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, xla, vulkan")
    parser.add_argument("--load", action="store_true", default=False) 
    parser.add_argument("--load_path", type=str, default=None, help="load path of model")
    parser.add_argument("--random_testing", action="store_true", default=False) 
    #parser.add_argument("--coef_target", type=int, default='auto', help="target for coeficient training")
    

    return parser.parse_args()

# python train.py --environment HopperPyBulletEnv-v0 --model clac --agent a1 --training_ts 1000 --device_type cuda 

# training time: 3M for all 
# ./train.sh a1 HopperPyBulletEnv-v0
# ./train.sh a1 AntPyBulletEnv-v0
# ./train.sh a1 HalfCheetahPyBulletEnv-v0
# ./train.sh a1 Walker2DPyBulletEnv-v0
# ./train.sh a1 ReacherPyBulletEnv-v0
# ./train.sh a1 PusherPyBulletEnv-v0

if __name__ == '__main__':
    arglist = parse_args()


    import time
    start_time = time.time()
    train(arglist)
    print("--- %s seconds ---" % (time.time() - start_time))

    
