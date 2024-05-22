""" Searches Lichess PGN data for classical chess games won by highly-rated 
    players as white, then encodes alternating moves as x and y training data """

import pickle
import chess.pgn
import io
from _0_chess_tools import fen_to_ascii, one_hot_encode
import numpy as np

ELO_LIMIT=1800

x_data = []
y_data = []

with open('training data/lichess_db_2014-04.pgn') as file:
    for index, line in enumerate(file):

        if 'WhiteElo' in line:
            elo_rating = ''.join([char for char in line if char.isnumeric()])
            if elo_rating == '':
                elo_rating = 0
            elo_rating = int(elo_rating)

        if 'Event' in line:
            if 'Classical' in line:
                classic = True
            else:
                classic = False

        if line[0:3] == '1. ' and classic and elo_rating > ELO_LIMIT and line[-1] == '0':

            pgn_moves = io.StringIO(line)
            game = chess.pgn.read_game(pgn_moves)
            board = game.board()

            for index, move in enumerate(game.mainline_moves()):
                board.push(move)
                ascii_board = fen_to_ascii(board.fen())
                tensor = one_hot_encode(ascii_board)
                flipped = np.flipud(tensor)    # play from black's perspective

                if index > 0 and index % 2 == 1:    
                    x_data.append(flipped)
                    
                elif index > 0 and index % 2 == 0:  
                    y_data.append(flipped)

            if len(x_data) == len(y_data) + 1:
                x_data.pop(-1)

print(len(x_data), len(y_data))

with open("training data/whole_game_data.pickle", "wb") as file:
    pickle.dump((x_data, y_data), file)

