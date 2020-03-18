"""
Based on the code from https://github.com/ajschumacher/ajschumacher.github.io/blob/master/20191103-q_learning_tic_tac_toe_briefly/q_learning_tic_tac_toe.ipynb

"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
print("Tensorflow", tf.__version__)
print("Num GPUs available:", len(tf.config.experimental.list_physical_devices('GPU')))

import random
import collections
import numpy as np


def new_board(size=3):
    return np.zeros(shape=(size, size))

def available_moves(board):
    return np.argwhere(board == 0)

def check_game_end(board):
    best = max(list(board.sum(axis=0)) +    # columns
               list(board.sum(axis=1)) +    # rows
               [board.trace()] +            # main diagonal
               [np.fliplr(board).trace()],  # other diagonal
               key=abs)
    if abs(best) == board.shape[0]:  # assumes square board
        return np.sign(best)  # winning player, +1 or -1
    if available_moves(board).size == 0:
        return 0  # a draw

def play(board, player_objs, epoch):
    if (epoch+1) % 100 == 0:
        print (epoch+1, ', ', sep = '', end = '', flush=True)
    
    for player in [1, -1]:
        player_objs[player].new_game()
    player = 1
    game_end = check_game_end(board)
    while game_end is None:
        move = player_objs[player].move(board, epoch)
        board[tuple(move)] = player
        game_end = check_game_end(board)
        player = -player  # switch players
    for player in [1, -1]:
        """
        Final rewards are:
        +1 for draw
        +2 for winning
        -2 for losing
        """
        if game_end == 0:
            final_reward = 1.
        else:
            final_reward = 2. if player == game_end else -2.
        player_objs[player].learn(final_reward)
    
    for player in [1, -1]:
        player_objs[player].soft_update()
    
    if (epoch+1) % 5000 == 0:
        checkpoint_file = './tic-tac-toe-{epoch:07d}'.format(epoch=epoch+1)
        for player in [1, -1]:
            player_objs[player].checkpoint(checkpoint_file)
    
    return game_end


from agent import Agent

from opponent import RandomAgent, SmartAgent

random.seed(11)

### Opponent making the first move
agent = Agent(size=3, seed=11)
agent.primary.summary()
agent.target.summary()
results = [play(new_board(), {+1: RandomAgent(), -1: agent}, epoch) for epoch in range(1000)]
agent.checkpoint('opponent')
collections.Counter(results)

agent.training = False
results_eval = [play(new_board(), {+1: RandomAgent(), -1: agent}, epoch) for epoch in range(1000)]
collections.Counter(results_eval)
results_eval = [play(new_board(), {+1: SmartAgent(+1, random_first=0), -1: agent}, epoch) for epoch in range(1000)]
collections.Counter(results_eval)
results_eval = [play(new_board(), {+1: SmartAgent(+1, random_first=1), -1: agent}, epoch) for epoch in range(1000)]
collections.Counter(results_eval)
results_eval = [play(new_board(), {+1: SmartAgent(+1, random_first=2), -1: agent}, epoch) for epoch in range(1000)]
collections.Counter(results_eval)


### AI agent making the first move
agent2 = Agent(size=3, seed=11)
agent2.primary.summary()
agent2.target.summary()
results = [play(new_board(), {+1: agent2, -1: RandomAgent()}, epoch) for epoch in range(1000)]
agent2.checkpoint('agent')
collections.Counter(results)

agent2.training = False
results_eval = [play(new_board(), {+1: agent2, -1: RandomAgent()}, epoch) for epoch in range(1000)]
collections.Counter(results_eval)
results_eval = [play(new_board(), {+1: agent2, -1: SmartAgent(-1, random_first=0)}, epoch) for epoch in range(1000)]
collections.Counter(results_eval)
results_eval = [play(new_board(), {+1: agent2, -1: SmartAgent(-1, random_first=1)}, epoch) for epoch in range(1000)]
collections.Counter(results_eval)
results_eval = [play(new_board(), {+1: agent2, -1: SmartAgent(-1, random_first=2)}, epoch) for epoch in range(1000)]
collections.Counter(results_eval)
