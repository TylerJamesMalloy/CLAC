import seaborn as sn 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 

envs = ["AntPyBulletEnv-v0", "HopperPyBulletEnv-v0", "HumanoidPyBulletEnv-v0", "ReacherPyBulletEnv-v0"]
agents = ["a1", "a2", "a3", "a4", "a5"]
models = ["clac", "sac"]

all_data = pd.DataFrame()

for env in envs:
    for agent in agents: 
        for model in models:
            data_path = "trained_models/" + env + "/" + model +"/normal/" + agent + "/data.pkl"
            if not os.path.exists(data_path): continue 
            data = pd.read_pickle(data_path)

            all_data = all_data.append(data, ignore_index=True)

print(all_data)

figs, axes = plt.subplots(4)

sn.lineplot(x="Timestep", y="Reward", data=all_data.loc[all_data['Environment'] == "AntPyBulletEnv-v0"], ax=axes[0], hue="Model")
sn.lineplot(x="Timestep", y="Reward", data=all_data.loc[all_data['Environment'] == "HopperPyBulletEnv-v0"], ax=axes[1], hue="Model")
sn.lineplot(x="Timestep", y="Reward", data=all_data.loc[all_data['Environment'] == "HumanoidPyBulletEnv-v0"], ax=axes[2], hue="Model")
sn.lineplot(x="Timestep", y="Reward", data=all_data.loc[all_data['Environment'] == "ReacherPyBulletEnv-v0"], ax=axes[3], hue="Model")

plt.show()