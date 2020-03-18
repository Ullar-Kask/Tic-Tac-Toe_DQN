import random
import math
import numpy as np
import tensorflow as tf

from model import TicTacToeModel


class Agent:
    """ NN player. """
    def __init__(self, size, seed, learning_rate=5e-3, eps_max=1., eps_min=5e-2, eps_decay=0.99993, tau=1e-3, gamma=0.99):
        self.size = size
        self.training = True
        self.learning_rate = learning_rate
        self.epsilon = eps_max
        self.eps_max = eps_max
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.tau = tau
        self.gamma = gamma
        self.primary = TicTacToeModel(size, seed)
        self.primary.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tf.losses.MeanSquaredError())
        self.primary.build(input_shape=(1, self.size*self.size)) # ready for model.summary()
        self.target = TicTacToeModel(size, seed)
        self.target.build(input_shape=(1, self.size*self.size)) # ready for model.summary()
        # Copy weights from primary network to target
        for t, p in zip(self.target.trainable_variables, self.primary.trainable_variables):
            t.assign(p)
    
    def new_game(self):
        self.last_move = None
        self.last_board = None
        self.last_q = None
    
    def get_move_index(self, epsilon, available_moves, max_q_index):
        """ Get epsilon-greedy move.
            Parameters:
                epsilon - epsilon
                available_moves - list of ravel indeces of available moves
                max_q_index - ravel index of the square with max Q-value
            Return:
                Index to array of ravel indeces of available moves
        """
        if np.random.random() < epsilon:
            move_index = np.random.choice(np.arange(len(available_moves)))
        else:
            aw = np.argwhere(np.asarray(available_moves) == max_q_index)
            move_index = aw[0,0]
        return move_index
    
    def move(self, board, step=0):
        q_primary = self.primary.predict(np.array([board.ravel()])).reshape(self.size, self.size)
        temp_q = q_primary.copy()
        temp_q[board != 0] = temp_q.min() - 1  # to avoid making move to an occupied cell
        max_q_index = np.argmax(temp_q)
        if not self.training:
            move = np.unravel_index(max_q_index, board.shape)
            return move
        """
        E.g. board is:
            [[0, 0, -1],
             [0, 1, 0],
             [0, 0, 0]]
        then available_moves is: [0, 1, 3, 5, 6, 7, 8]
        """
        available_moves = [np.ravel_multi_index(i, board.shape) for i in np.argwhere(board == 0)]
        move_index = self.get_move_index(self.epsilon, available_moves, max_q_index)
        move = np.unravel_index(available_moves[move_index], board.shape)
        if self.last_move is not None:
            q_target = self.target.predict(np.array([board.ravel()])).reshape(self.size, self.size)
            q_value = self.gamma*q_target[self.last_move]
            self.learn(q_value)
        self.last_board = board.copy()
        self.last_q = q_primary
        self.last_move = move
        self.epsilon = max(self.eps_min, self.eps_decay*self.epsilon)
        return move
    
    def learn(self, reward_value):
        if not self.training:
            return
        last_q = self.last_q.copy()
        last_q[self.last_move] = reward_value
        self.primary.fit(np.array([self.last_board.ravel()]), np.array([last_q.ravel()]), verbose=0)
    
    def soft_update(self):
        for t, p in zip(self.target.trainable_variables, self.primary.trainable_variables):
            t.assign(t * (1 - self.tau) + p * self.tau)
    
    def checkpoint(self, checkpoint_file):
        self.primary.save_weights(checkpoint_file+'.ckpt')
        self.target.save_weights(checkpoint_file+'_target.ckpt')

