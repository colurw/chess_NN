""" Takes the parsed FENs and converts them into one-hot tensors.  The one-hot tensors are an array
with 64 rows (each represents a square on the chess board) and 13 columns (each represents a possible 
piece that can occupy a square).  These are a sparse data format, containing mostly zeroes. 

Applying the first move in the lichess data creates the x_data, then applying the second move creates 
the y_data """

import csv
import numpy as np
import pickle
from _0_chess_tools import one_hot_encode

# Convert FENs from parser.py to ASCII boards, apply moves, then save to lists of one-hot encoded tensors
all_puzzles = []
all_solutions = []
with open('training data/allPuzzles_conv_w.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    # Create ASCII board
    for row in reader:
        FEN = row[1]
        board = ''
        for character in FEN:
            if character.isalpha():
                board = board + ' ' + character  
            if character.isnumeric():
                for n in range(int(character)):
                    board = board + ' ' + '.' 
            if character == '/':
                continue
            if character == ' ':
                break
        list = board.split()
        arr = np.array(list)
        arr = arr.reshape(-1, 8)
        
        # Get moves and strip last rank promotion code
        moves_qnk = str(row[2])    
        dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
        moves = moves_qnk.translate({ord(i): None for i in 'qnkr'})   # 'e5f6q e8e1 g1f2 e1f1' -> 'e5f6 e8e1 g1f2 e1f1'

        # Play first black move to create puzzle
        x_from = dict[moves[0]]
        y_from = 8 - int(moves[1])
        x_to = dict[moves[2]]
        y_to = 8 - int(moves[3])
        piece = arr[y_from][x_from]
        arr[y_from][x_from] = '.'
        arr[y_to][x_to] = piece
        arr_flipped = np.fliplr(arr)
        #arr_flipped = np.flipud(arr_flipped)    # omit this line when allPuzzles_conv_w.csv loaded in line 16
        all_puzzles.append(one_hot_encode(arr_flipped))    

        # Play white response to create solution
        try:
            x_from = dict[moves[5]]
            y_from = 8-int(moves[6])
            x_to = dict[moves[7]]
            y_to = 8 - int(moves[8])
        except:
            print('FAIL!! ', moves)          
        piece = arr[y_from][x_from]
        arr[y_from][x_from] = '.'
        arr[y_to][x_to] = piece
        arr_flipped = np.fliplr(arr)   
        #arr_flipped = np.flipud(arr_flipped)    # omit this line when allPuzzles_conv_w.csv loaded in line 16
        all_solutions.append(one_hot_encode(arr_flipped))

print(len(all_solutions), 'puzzles encoded')

with open("training data/all_puzz_conv_w.pickle", "wb") as file:
    pickle.dump((all_puzzles), file)

with open("training data/all_solns_conv_w.pickle", "wb") as file:
    pickle.dump((all_solutions), file)