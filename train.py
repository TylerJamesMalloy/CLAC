import gym
import pybulletgym
import numpy as np
import pandas as pd 
import argparse
import time

from torch._C import device

from clac import CLAC
from stable_baselines3 import SAC


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

    env.unwrapped.randomize() 
    return np.mean(episode_rewards)

def train(args):
    env = gym.make(arglist.environment)

    if(arglist.model == "clac"):
        model = CLAC("MlpPolicy", env, verbose=1, target_mutual_information="auto", device=arglist.device_type)
    elif(arglist.model == "sac"):
        model = SAC("MlpPolicy", env, verbose=1, target_entropy="auto", device=arglist.device_type)
    
    rewardsDataFrame = pd.DataFrame()
    training_type = 'randomized' if arglist.random_training else 'normal' # not implemented yet 
    save_folder = "./trained_models/" + arglist.environment + "/" + arglist.model + "/" + training_type + "/" + arglist.agent + "/" # each agent gets its own folder 
    
    if not arglist.load:
        training_timestep = 0
        for _ in range(arglist.training_evals):
            start_time = time.time()
            model.learn(total_timesteps= arglist.training_ts / arglist.training_evals, log_interval= arglist.training_ts / arglist.training_evals)
            training_timestep += arglist.training_ts / arglist.training_evals
            mean_reward = evaluate(model, env, arglist)
            print("Mean reward: ", mean_reward, " at training timestep ", training_timestep, " took ", (time.time() - start_time), " seconds") # add time 

            rewardDataPoint = {"Timestep": training_timestep , "Reward": mean_reward, "Model": arglist.model, "Environment": arglist.environment, "Training": training_type}
            rewardsDataFrame = rewardsDataFrame.append(rewardDataPoint, ignore_index=True)

            if arglist.random_training: 
                env.unwrapped.randomize()

        model.save(save_folder + "/" + arglist.agent)
    else:
        model = model.load(arglist.load_path)

        mean_reward = evaluate(model, env, arglist)
        rewardDataPoint = {"Episode": arglist.training_ts , "Reward": mean_reward, "Model": arglist.model, "Environment": arglist.environment, "Training": training_type}
        rewardsDataFrame = rewardsDataFrame.append(rewardDataPoint, ignore_index=True)

    rewardsDataFrame.to_pickle(save_folder + "/data.pkl")
    print(rewardsDataFrame)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--environment", type=str, default="Pendulum-v0", help="training environment to train on")
    parser.add_argument("--model", type=str, default="clac", help="model type to train on")
    parser.add_argument("--agent", type=str, default="a1", help="name of agent, for saving results")
    parser.add_argument("--training_ts", type=int, default=1000000, help="number of time steps to train on")
    parser.add_argument("--training_evals", type=int, default=1000, help="number of time steps to train on")
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

# python train.py --environment ReacherPyBulletEnv-v0 --model sac --agent a1 --training_ts 1000000 --device_type cuda 

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

    
