import random
import numpy as np


class RandomAgent:
    """ Player making random moves. """
    """ Used to play against neural network during training. """
    def new_game(self):
        pass
    def learn(self, value):
        pass
    def soft_update(self):
        pass
    def checkpoint(self, checkpoint_file):
        pass
    def move(self, board, step=0):
        return random.choice(np.argwhere(board == 0))


class SmartAgent:
    """ Used to play against neural network during training. """
    def __init__(self, player, random_first=1):
        """ random_first=0 - first move to the center square
            random_first=1 - first move to the center or a corner square with 20% probabilities each
            random_first=2 - first move totally random
        """
        self.player = player
        self.random_first = random_first
    
    def new_game(self):
        pass
    def learn(self, value):
        pass
    def soft_update(self):
        pass
    def checkpoint(self, checkpoint_file):
        pass
    
    def blocking_move(self,board,player):
        # Check if 2 pieces in a row along columns
        bs=board.sum(axis=0)
        amax=np.argmax(bs) if player==1 else np.argmin(bs)
        if bs[amax] == 2.0*player:
            amin = np.argmin(board,axis=0) if player==1 else np.argmax(board,axis=0)
            return np.asarray([amin[amax],amax])
        # Check if 2 pieces in a row along rows
        bs=board.sum(axis=1)
        amax=np.argmax(bs) if player==1 else np.argmin(bs)
        if bs[amax] == 2.0*player:
            amin = np.argmin(board,axis=1) if player==1 else np.argmax(board,axis=1)
            return np.asarray([amax,amin[amax]])
        # Check if 2 pieces in a row along main diagonal
        if board.trace() == 2.0*player:
            for i in range(3):
                if board[i,i] == 0:
                    return np.asarray([i,i])
        # Check if 2 pieces in a row along secondary diagonal
        b=np.fliplr(board)
        if b.trace() == 2.0*player:
            for i in range(3):
                if b[i,i] == 0:
                    return np.asarray([i,2-i])
        return None
    
    def find_move(self, board):
        if np.count_nonzero(board) == 0:
            if self.random_first == 0:
                return np.asarray([1,1])
            elif self.random_first == 1:
                c = random.random()
                if c < 0.2:
                    return np.asarray([1,1])
                elif c < 0.4:
                    return np.asarray([0,0])
                elif c < 0.6:
                    return np.asarray([0,2])
                elif c < 0.8:
                    return np.asarray([2,0])
                else:
                    return np.asarray([2,2])
            elif self.random_first == 2:
                return None
        
        # Check if this party has a winning move
        y = self.blocking_move(board, self.player)
        if y is not None:
            return y
        
        # Check if the other party has a winning move
        y = self.blocking_move(board, -self.player)
        if y is not None:
            return y
        
        if board[1,1] == 0:
            return np.asarray([1,1])
        
        if board[0,0] == 0 and board[0,2] == 0 and board[2,0] == 0 and board[2,2] == 0:
            c = random.random()
            if c < 0.25:
                return np.asarray([0,0])
            elif c < 0.5:
                return np.asarray([0,2])
            elif c < 0.75:
                return np.asarray([2,0])
            else:
                return np.asarray([2,2])
        elif board[0,0] == 0 and board[0,2] == 0 and board[2,0] == 0:
            c = random.random()
            if c < 1./3.:
                return np.asarray([0,0])
            elif c < 2./3.:
                return np.asarray([0,2])
            else:
                return np.asarray([2,0])
        elif board[0,0] == 0 and board[0,2] == 0 and board[2,2] == 0:
            c = random.random()
            if c < 1./3.:
                return np.asarray([0,0])
            elif c < 2./3.:
                return np.asarray([0,2])
            else:
                return np.asarray([2,2])
        elif board[0,0] == 0 and board[2,0] == 0 and board[2,2] == 0:
            c = random.random()
            if c < 1./3.:
                return np.asarray([0,0])
            elif c < 2./3.:
                return np.asarray([2,0])
            else:
                return np.asarray([2,2])
        elif board[0,2] == 0 and board[2,0] == 0 and board[2,2] == 0:
            c = random.random()
            if c < 1./3.:
                return np.asarray([0,2])
            elif c < 2./3.:
                return np.asarray([2,0])
            else:
                return np.asarray([2,2])
        elif board[0,0] == 0 and board[0,2] == 0:
            c = random.random()
            if c < 0.5:
                return np.asarray([0,0])
            else:
                return np.asarray([0,2])
        elif board[0,0] == 0 and board[2,0] == 0:
            c = random.random()
            if c < 0.5:
                return np.asarray([0,0])
            else:
                return np.asarray([2,0])
        elif board[0,0] == 0 and board[2,2] == 0:
            c = random.random()
            if c < 0.5:
                return np.asarray([0,0])
            else:
                return np.asarray([2,2])
        elif board[0,2] == 0 and board[2,0] == 0:
            c = random.random()
            if c < 0.5:
                return np.asarray([0,2])
            else:
                return np.asarray([2,0])
        elif board[0,2] == 0 and board[2,2] == 0:
            c = random.random()
            if c < 0.5:
                return np.asarray([0,2])
            else:
                return np.asarray([2,2])
        elif board[2,0] == 0 and board[2,2] == 0:
            c = random.random()
            if c < 0.5:
                return np.asarray([2,0])
            else:
                return np.asarray([2,2])
        elif board[0,0] == 0:
            return np.asarray([0,0])
        elif board[0,2] == 0:
            return np.asarray([0,2])
        elif board[2,0] == 0:
            return np.asarray([2,0])
        elif board[2,2] == 0:
            return np.asarray([2,2])
        
        return None
    
    def move(self, board, step=0):
        m = self.find_move(board)
        if m is not None:
            return m
        else:
            return random.choice(np.argwhere(board == 0))
