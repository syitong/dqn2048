#This is the game "2048".

import copy
from board import Board
from model import dqn_agent
import numpy as np
import sys

class Game:
    """
    this class provides interfaces of playing games by
    human players or ai players, recording games, saving
    games and replaying games.
    """
    def __init__(self):
        self.pow = './game_archives/'
        self.agent_path = 'trained_agents/'
        self.idle()

    def idle(self):
        self._game = list()
        order = input(
"""Choose one from the list:
    1: new game
    2: new game by ai
    3: new game by ai quiet
    4: load
    5: replay
    6: auto train ai
    7: generate data by search
    0: exit
    """)
        if order == '1':
            parameter = {'size': eval(input('size: ')),
                'odd_2': eval(input('odd of 2(between 0 and 1): '))}
            self.new_game(parameter)
        if order == '2':
            parameter = {'size': eval(input('size: ')),
                'odd_2': eval(input('odd of 2(between 0 and 1): '))}
            name = input('Name of the ai player: ')
            self.new_game_by_ai(name,parameter,0)
        if order == '3':
            parameter = {'size': eval(input('size: ')),
                'odd_2': eval(input('odd of 2(between 0 and 1): '))}
            name = input('Name of the ai player: ')
            self.new_game_by_ai(name,parameter)
        if order == '4':
            self.load(input('Name of the game: '))
        if order == '5':
            self.replay(input('Name of the game: '))
        if order == '6':
            self.auto_train_ai()
        if order == '7':
            parameter = {'size': eval(input('size: ')),
                'odd_2': eval(input('odd of 2(between 0 and 1): '))}
            self.gen_search_data(parameter)
        if order == '0':
            pass

    def new_game(self,parameter):
        self._board = Board(parameter)
        self._play()

    def _play(self):
        endgame_flag = self._board.gameend()
        while endgame_flag == 0:
            self._board.print_board()
            self.push()
            order = input('Your Move(wsad,e:exit): ')
            while order != 'e' and self._board.move(order) == -1:
                order = input('Your Move(wsad,e:exit): ')
                if order == 'e':
                    break
            if order == 'e':
                break
            self.push(order)
            self._board.next()
            endgame_flag = self._board.gameend()

        if endgame_flag == 1:
            self._board.print_board()
            print('Game over!')
        if input('Do you want to save the game?(y/n) ') == 'y':
            self.save(input('Name of the game: '))
        self.idle()

    def save(self,filename):
        with open(self.pow+filename,'w') as f:
            for line in self._game:
                f.write(str(line))
                f.write('\n')
        with open(self.pow+filename+'.para','w') as f:
            f.write(str(self._board.para))

    def _load_game(self,filename):
        with open(self.pow+filename,'r') as f:
            for line in f:
                self._game.append(eval(line))
        with open(self.pow+filename+'.para','r') as f:
            self._board = Board(eval(f.read()))

    def load(self,filename):
        self._load_game(filename)
        self.pop()
        self._play()

    def push(self,move=None):
        temp = []
        if move==None:
            for row in self._board:
                temp.extend(row)
            self._game.append(temp)
        else:
            self._game[-1].append(move)

    def pop(self):
        temp_board = list()
        n = self._board.para['size']
        temp_row = self._game.pop()
        for idx in range(n):
            temp_board.append(temp_row[idx*n:(idx+1)*n])
        self._board.load_board(temp_board)

    def read(self,idx):
        temp_board = list()
        n = self._board.para['size']
        temp_row = self._game[idx]
        for jdx in range(n):
            temp_board.append(temp_row[jdx*n:(jdx+1)*n])
        self._board.load_board(temp_board)
        if len(temp_row) == n*n+1:
            return self._game[idx][-1]
        else:
            return None

    def new_game_by_ai(self,name,parameter,quiet=1):
        self._board = Board(parameter)
        player = dqn_agent({'name':name}, load=True)
        endgame_flag = self._board.gameend()
        while endgame_flag == 0:
            self._board.print_board()
            self.push()
            if quiet != 1:
                if input('next?(y/n) ') == 'n':
                    break
            board = np.ma.log2(self._board).filled(0.)
            order = player.play_action(board)
            if self._board.move(order) == -1:
                print('AI error.')
                break
            self.push(order)
            self._board.next()
            endgame_flag = self._board.gameend()
        if endgame_flag == 1:
            self._board.print_board()
            print('Game over!')
        if input('Do you want to save the game?(y/n) ') == 'y':
            self.save(input('Name of the game: '))
        self.idle()

    # def gen_search_data(self,parameter):
    #     player = Ai()
    #     player.new('ss')
    #     rounds = eval(input('Number of rounds: '))
    #     for idx in range(rounds):
    #         print('Round: ',idx)
    #         self._board = Board(parameter)
    #         endgame_flag = self._board.gameend()
    #         while endgame_flag == 0:
    #             self.push()
    #             board = copy.deepcopy(self._board)
    #             move = player.simple_search(board)
    #             if self._board.move(move) == 0:
    #                 print('AI error.')
    #                 break
    #             self.push(move)
    #             self._board.next()
    #             endgame_flag = self._board.gameend()
    #         if endgame_flag == 1:
    #             print('Game over!')
    #     if input('Do you want to save the game?(y/n) ') == 'y':
    #         self.save(input('Name of the game: '))
    #     self.idle()

    def replay(self,filename):
        self._load_game(filename)
        for idx in range(len(self._game)):
            move = self.read(idx)
            self._board.print_board()
            if move == None:
                print('Game over!')
                break
            print(move)
            if input('next?(y/n) ') == 'y':
                self._board.move(move)
            else:
                break
        self.idle()

    def auto_train_ai(self):
        name = input('Name of the AI: ')
        try:
            with open(self.agent_path + name + '-params','rb') as f:
                pass
        except:
            print('New agent!')
            params = {'name':name,
                'N':10000,
                'shape':[4,4],
                'ep_start':1.,
                'ep_end':0.01,
                'ep_rate':0.001,
                'batch_size':eval(input('Batch size: ')),
                'a_list':['w','s','a','d'],
                'C':1000,
                'lrate':0.001,
            }
            player = dqn_agent(params)
        else:
            print('Find saved agent with the same name.')
            if input('Do you want to 1. load it, 2. overwrite it?') == '1':
                params = {'name': name}
                player = dqn_agent(params, load = True)
            else:
                params = {'name':name,
                    'N':10000,
                    'shape':[4,4],
                    'ep_start':1.,
                    'ep_end':0.1,
                    'ep_rate':0.001,
                    'batch_size':eval(input('Batch size: ')),
                    'a_list':['w','s','a','d'],
                    'C':1000,
                    'lrate':0.001,
                }
                player = dqn_agent(params)
        rounds = eval(input('Number of rounds: '))
        best = 0
        for idx in range(rounds):
            self._game = list()
            self._board = Board(player.game_para)
            endgame_flag = self._board.gameend()
            while endgame_flag == 0:
                # self.push()
                board = copy.deepcopy(self._board)
                order = player.train_action(board)
                rew = self._board.move(order, quiet=1)
                board_nxt = copy.deepcopy(self._board)
                endgame_flag = self._board.gameend()
                if rew > 0:
                    self._board.next()
                else:
                    endgame_flag = 1
                # self.push(order)
                if endgame_flag:
                    total = np.sum(self._board)
                    if total > best:
                        best = total
                    print('Total Score: {}        '.format(
                        total))
                obs = {
                    's': np.ma.log2(board).filled(0),
                    'a': order,
                    'r': rew,
                    'ss': np.ma.log2(board_nxt).filled(0),
                    'done': endgame_flag
                }
                player.perceive(obs)
                player.train()
        print('\nBest Score: ' + str(best))
        player.save()
        self.idle()


#The following is the main program.

if __name__ == '__main__':
    game = Game()
