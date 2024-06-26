import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.01

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #something to do with randomness
        self.gamma = 0.8  #discount rate? has to be < 1; also can be changed;
        self.memory = deque(maxlen = MAX_MEMORY) #pops elements from left automatically, after max memory size is reached by the agent;
        self.model = Linear_QNet(11, 256, 3) #input and output are fixed as per our program, but number of hidden layers can be messed around with;
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def  get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
    

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        
        return np.array(state, dtype = int) #nice lil conversion from bool t, f to 0, 1 int



    def remember(self, state, action, reward, next_state, finished):
        self.memory.append((state, action, reward, next_state, finished)) #pops from left if size exceeds given size;

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #returns random tuples from memory, of size = batch_size
        else:
            mini_sample = self.memory

        states, action, rewards, next_states, dones = zip(*mini_sample) #short way of separating all dones, all next states, all actions, etc etc

    def train_short_memory(self, state, action, reward, next_state, finished):
        self.trainer.train_step(state, action, reward, next_state, finished)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation; explore first, then learn and decrease random exploration
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_score = []

    total_score = 0
    best_score = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        #get current state
        old_state = agent.get_state(game)

        final_move = agent.get_action(old_state)

        reward, done, score = game.play_step(final_move)

        new_state = agent.get_state(game)

        #train the short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        
        #remember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            #train permanent memory; long term memory for future games;
            #also plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)


if __name__ == '__main__':
    train()

    
