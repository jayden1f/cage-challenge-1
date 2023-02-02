import inspect
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from statistics import mean, stdev
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.SkyNet import SkyNetBase
from CybORG.Agents.SimpleAgents.BlueLoadAgent import BlueLoadAgent
from CybORG.Agents.SimpleAgents.BlueReactAgent import BlueReactRemoveAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper
import numpy as np

MAX_EPS = 10
agent_name = 'Blue'
linePrint = 1
totalAvg = 0
totalStd = 0

def wrap(env):
    return OpenAIGymWrapper(agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))


if __name__ == "__main__":
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    # ask for a name
    name = input('Name: ')
    # ask for a team
    team = input("Team: ")
    # ask for a name for the agent
    name_of_agent = input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    agent = SkyNetBase()

    print(f'Using agent {agent.__class__.__name__}\n')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{1.0}, {scenario}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    if linePrint == 1:
        print("{:<15} {:<25} {:<20} {:<10}".format('Agent','Score','Standard Deviation','Step'))
        linePrint - 1
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().o112bservation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            obs = [1,11293]
            obs = np.empty(obs)
            for i in range(MAX_EPS):
                r = []
                a = []
                for j in range(num_steps):
                    action, predictions = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                    obs = observation
                total_reward.append(sum(r))
                actions.append(a)
                observation = wrapped_cyborg.reset()
            print(90*'-')
            print ("{:<15} {:<25} {:<20} {:<10}".format(red_agent.__name__ , mean(total_reward) , stdev(total_reward)  , num_steps))
            totalAvg = totalAvg + mean(total_reward)
            totalStd = totalStd + stdev(total_reward)
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
    print(90*'-')
    print("Total Score:", totalAvg, "+-", totalStd)
    print('\n')

