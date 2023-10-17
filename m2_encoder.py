import csv
import numpy as np
import pickle
from _0_chess_tools import one_hot_encode

# Convert FENs to 2D boards, apply moves, save to lists of one-hot encoded tensors
mate_in_two = []
mate_in_one = []
solution = []
with open('mateIn2_white.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
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
        
        # Get moves
        moves = str(row[2])  ##'e5f6 e8e1 g1f2 e1f1'
        dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
        
        # Play first black move to create puzzle
        x_from = dict[moves[0]]
        y_from = 8-int(moves[1])
        x_to = dict[moves[2]]
        y_to = 8 - int(moves[3])
        piece = arr[y_from][x_from]
        arr[y_from][x_from] = '.'
        arr[y_to][x_to] = piece
        mate_in_two.append(one_hot_encode(arr))
        mirror = np.fliplr(arr)
        mate_in_two.append(one_hot_encode(mirror))
        mirror2 = np.flipud(arr)
        mate_in_two.append(one_hot_encode(mirror2))
        mirror3 = np.fliplr(mirror2)
        mate_in_two.append(one_hot_encode(mirror3))
        
        # Play first white move
        x_from = dict[moves[5]]
        y_from = 8-int(moves[6])
        x_to = dict[moves[7]]
        y_to = 8 - int(moves[8])
        piece = arr[y_from][x_from]
        arr[y_from][x_from] = '.'
        arr[y_to][x_to] = piece
        
        # Play second black move
        x_from = dict[moves[10]]
        y_from = 8-int(moves[11])
        x_to = dict[moves[12]]
        y_to = 8 - int(moves[13])
        piece = arr[y_from][x_from]
        arr[y_from][x_from] = '.'
        arr[y_to][x_to] = piece
        mate_in_one.append(one_hot_encode(arr))
        mirror = np.fliplr(arr)
        mate_in_one.append(one_hot_encode(mirror))
        mirror2 = np.flipud(arr)
        mate_in_one.append(one_hot_encode(mirror2))
        mirror3 = np.fliplr(mirror2)
        mate_in_one.append(one_hot_encode(mirror3))
        
        # Show checkmate position 
        x_from = dict[moves[15]]
        y_from = 8 - int(moves[16])
        x_to = dict[moves[17]]
        y_to = 8 - int(moves[18])
        piece = arr[y_from][x_from]
        arr[y_from][x_from] = '.'
        arr[y_to][x_to] = piece
        solution.append(one_hot_encode(arr))
        mirror = np.fliplr(arr)
        solution.append(one_hot_encode(mirror))
        mirror2 = np.flipud(arr)
        solution.append(one_hot_encode(mirror2))
        mirror3 = np.fliplr(mirror2)
        solution.append(one_hot_encode(mirror3))

print(len(solution), 'puzzles encoded')

with open("mate_in_2.pickle", "wb") as file:
    pickle.dump((mate_in_two), file)

with open("mate_in_1.pickle", "wb") as file:
    pickle.dump((mate_in_one), file)

with open("solution.pickle", "wb") as file:
    pickle.dump((solution), file)

