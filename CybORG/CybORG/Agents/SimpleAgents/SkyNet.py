'''
This is a modified version of the CybORG BlueLoadAgent that was developed to enhance the capability of the original and improve on it

Developed:  Sky TianYi Zhang
            Mitchell Knyn
            Jayden Fowler
            
Reference: pythonlessons https://pylessons.com/LunarLander-v2-PPO

Last Modified: 22 Feb 2023
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #Suppress warning messages
import copy
import pylab
import numpy as np
import inspect
import tensorflow as tf
tf.config.run_functions_eagerly(True)     #Suppress warning messages
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorboardX import SummaryWriter 
from CybORG import CybORG
from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents import B_lineAgent, SleepAgent, RedMeanderAgent #Import all three red agent 

critic_ppo_path = str(inspect.getfile(CybORG)) #Used to find ppo file path ignoring user's name and directory
critic_ppo_path = critic_ppo_path[:-10] + "/Evaluation/" #Under the evaluation folder
actor_ppo_path = str(inspect.getfile(CybORG)) 
actor_ppo_path = actor_ppo_path[:-10] + "/Evaluation/"
path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'  #Find the Scenario1b.yaml and locate it's path
env = (CybORG(path, 'sim')) #Load the Scenario1b.yaml into the environment
cyborg = OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env)))) #Use wrapper to wrap the environment into cyborg environment

class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape) #Define input shape
        self.action_space = action_space #Total number of possible actions avaiable in the current environment, which is 54 (blue agent)
        #Three layers of neural network with 512, 256, 64
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X) #Classification layer 
        output = Dense(self.action_space, activation="softmax")(X) #Total number of output nodes of the neural network

        self.Actor = Model(inputs=X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr)) #learn and correct actions of the agent

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2 #Customised loss setting it to maximum percent 20% only
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2)) #Calculate by getting a negative mean of element wise minimum value of p1 and p2

        return actor_loss

    def gaussian_likelihood(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.Actor.predict(state)

class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        V = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)
        V = Dense(256, activation="relu", kernel_initializer='he_uniform')(V)
        V = Dense(64, activation="relu", kernel_initializer='he_uniform')(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

class SkyNetBase(BaseAgent):
    def __init__(self):
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml' 
        #env = (CybORG(path, 'sim')) #Comment this line when using evaluation.py
        env = (CybORG(path, 'sim',agents={'Red': B_lineAgent})) #Comment this line when using SkyNet.py to train
        cyborg = OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))
        self.env = cyborg
        self.env_name = 'CybORG'
        self.action_size = self.env.get_action_space('Blue')
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 100 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle=False
        self.Training_batch = 1000
        self.optimizer = Adam
        self.x_value = 1
        self.y_value = 1
        self.goes = 10  #10 is max before crashing
        self.Actor_name = "PPO_Actor.h5"
        self.Critic_name = "PPO_Critic.h5"
        self.correct = 0
        
        self.replay_count = 0
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.save()

        # Create Actor-Critic network models
        if os.path.exists(actor_ppo_path):
            self.load()
            print("\nLoaded Training Models\n")
        
    def load(self): #Load weights
        self.Actor.Actor.load_weights(critic_ppo_path+f"{self.Actor_name}")
        self.Critic.Critic.load_weights(actor_ppo_path+f"{self.Critic_name}")

    def save(self): #Save weights
        self.Actor.Actor.save_weights(critic_ppo_path+f"{self.Actor_name}",self.Actor_name)
        self.Critic.Critic.save_weights(critic_ppo_path+f"{self.Critic_name}",self.Critic_name)
        
    def replay(self, states, actions, rewards, predictions, dones, next_states, score):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        advantages.resize(self.x_value,54)
        predictions.resize(self.correct,54)
        actions.resize(self.correct,54)
        
        y_true = np.hstack([advantages, predictions, actions])
        target.resize(self.y_value,self.y_value)
        self.y_value += 1
        values.resize(self.episode,self.episode)
        
        # training Actor and Critic networks
        if score == 0.0:
            a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
            c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
            self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
            self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1
        self.x_value += 1
        
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        r = rewards
        nv = next_values
        v = values
        if dones == False:
            d = 0
        else:
            d = 1
        deltas = [r + gamma * (1 - d) * nv - v] 
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)
        
    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = self.Actor.predict(state)
        action = np.random.choice(self.action_size, p=prediction)
   
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction
        
    def run(self): # train only when episode is finished
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        temp = 0
        total_score=0
        total_dev=0
        print("Round:", temp)
        while done == False:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            if temp >= self.goes:
                done = True
                self.env.close() 
                break
            else:
                self.load() #Load actors weights
                while self.episode < self.EPISODES:
                    #self.env.render()
                    # Actor picks an action
                    action, action_onehot, prediction = self.act(state)
                    # Retrieve new state, reward, and whether the state is terminal
                    next_state, reward, done, _ = self.env.step(action)
                    # Memorize (state, action, reward, done) for training
                    states.append(state)
                    next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                    actions.append(action_onehot)
                    rewards.append(reward)
                    dones.append(done)
                    predictions.append(prediction)
                    # Update current state
                    state = np.reshape(next_state, [1, self.state_size[0]])
                    #Incrememt scores and episode
                    score += reward
                    self.episode += 1
                    self.correct += 1
                    average = self.PlotModel(score, self.episode)
                    print("Round: {}, episode: {}/{}, score: {}, average: {:.2f}".format(temp, self.episode, self.EPISODES, score, average))
                    total_score += score
                    total_dev += average
                    self.save() #Save actor weight
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/lr', self.lr, self.episode)
                    
                    self.replay(states, actions, rewards, predictions, dones, next_states, score)

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape(state, [1, self.state_size[0]])

                if self.episode >= self.EPISODES:
                    temp += 1
                    self.episode = 0
                    self.x_value = 1
                    self.y_value = 1
                    self.save()
                    self.env.reset()
                    if temp == self.goes:
                        print("Round:", temp,"Steps:", self.EPISODES*self.goes, "Final Score:", total_score, "Deviation:", total_dev)    
        
    def PlotModel(self, score, episode): #Graph of the steps result
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            #pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.title(self.env_name + " PPO training cycle", fontsize=18)
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")#Save the training graph result
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average:
            self.max_average = self.average_[-1]
            self.save()
            # decrease learning rate every saved model
            self.lr *= 0.95
            K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)

        return self.average_[-1]
        
    def get_action(self,state): #Get the next predicted action and return it
        state = np.reshape(state,(1,11293)) #Create the same sized state
        action = self.Actor.predict(state)
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction) #Choose an action
        return action, prediction
            
    def test(self, test_episodes = 100):
        self.load()
        e = 0
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done = False
        score = 0
        while not done:
            action = np.argmax(self.Actor.predict(state)[0])
            state, reward, done, _ = self.env.step(action)
            state = np.reshape(state, [1, self.state_size[0]])
            score += reward
            print("episode: {}/{}, score: {}".format(e, 100, score))
            if e >= 100:
                done = True
            e += 1
        self.env.close()
        
if __name__ == "__main__":
    agent = SkyNetBase()
    agent.run()
    #agent.test(agent) # train as PPO, train every episode
