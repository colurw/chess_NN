""" Searches Lichess PGN data for classical chess games won by highly-rated 
    players as white, then encodes alternating moves as x and y training data.
    https://database.lichess.org/ """

import chess.pgn
import io
from _0_chess_tools import fen_to_ascii, one_hot_encode
import numpy as np
import zstandard as zstd

MINIMUM_ELO=1900 

x_data = []
y_data = []
count = 0

with zstd.open('training data/lichess_db_standard_rated_2016-08.pgn.zst', 'r') as file:
    for i, line in enumerate(file):

        if 'Event' in line:
            if 'Classical' in line:
                classic = True
            else:
                classic = False

        elif 'WhiteElo' in line:
            elo_rating = ''.join([char for char in line if char.isnumeric()])
            if elo_rating == '':
                elo_rating = 0
            elo_rating = int(elo_rating)

        elif 'Termination' in line:
            if 'Normal' in line:
                normal = True
            else:
                normal = False      

        # if line contains moves, all priors are true, and white wins, load game into python-chess
        elif line[0:3] == '1. ' and classic and elo_rating > MINIMUM_ELO and normal and '0' in line[-2]:
            pgn_moves = io.StringIO(line)
            game = chess.pgn.read_game(pgn_moves)
            board = game.board()

            # create one-hot training data
            for index, move in enumerate(game.mainline_moves()):
                board.push(move)
                ascii_board = fen_to_ascii(board.fen())
                tensor = one_hot_encode(ascii_board)
                # ...from black's perspective
                flipped = np.flipud(tensor)    

                if index > 0 and index % 2 == 1:    
                    x_data.append(flipped)
                    
                elif index > 0 and index % 2 == 0:  
                    y_data.append(flipped)

            if len(x_data) == len(y_data) + 1:
                x_data.pop(-1)

            count += 1

print(count, 'games encoded')
print(len(x_data), len(y_data), 'moves encoded')

np.save('whole_game_x.npy', x_data)
np.save('whole_game_y.npy', y_data)


