# This file is not being used.
# All functions have been transferred into the main file calling the BcaEnv.
# Because the agent is part of the main file

import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from helper_plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    # transferred
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness, greedy/exploration
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if memory larger, it calls popleft()
        self.model = Linear_QNet(11,256,3) # neural network
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    # need to adjust to save states and memory cache to include next state or add current state to previous memory chunk
    def get_state(self, game): #observation in BcaEnv
        pass

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) #deque will popleft() if max_memory is reached. Extra () parantheses to store values as single tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # return list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*mini_sample) # unpack into lists rather than combined tuples
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)
    
    # transfered, loosely
    def get_action(self, state): #action/actuation in BcaEnv
        # random moves: exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = None # [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move





def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI() # BcaEnv?? TODO
    while True:
        #get old state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train long memory - also called replay memory or experience replay
            # plot results
            game.reset
            agent.n_games += 1
            agent.train_long_memory

            if score < record:
                record = score
                agent.model.save()

            print('Game ', agent.n_games, 'Score ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)





if __name__ == '__main__':
    train()

