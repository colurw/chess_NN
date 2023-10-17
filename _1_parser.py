""" Reads Forsyth-Edwards notation 'find the best move' chess puzzles from lichess.com database.  
Data is cleaned and parsed according to puzzle-type.  Black-to-play games converted into 
white-to-play for consistency when training neural network. """

import csv
import itertools

# Parse play as black puzzles from lichess database
count = 0
with open('training data/chess_puzzles.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in itertools.islice(reader, 3000000):
        if str(' w ') in str(row[1]):                                   # only get puzzles played as black 
            if str(row[2][5]) != ' ' and str('q') not in str(row[2]):   # ignore puzzles with queening 
                count = count + 1
                with open('training data/allPuzzles_b.csv', 'a', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(row[:4])
    print(count)

# Parse play as white puzzles from lichess database
count = 0
with open('training data/chess_puzzles.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in itertools.islice(reader, 3000000):
        if str(' w ') not in str(row[1]):                               # only get puzzles played as white 
            if str(row[2][5]) != ' ' and str('q') not in str(row[2]):   # ignore puzzles with queening 
                count = count + 1
                with open('training data/allPuzzles_w.csv', 'a', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(row[:4])
    print(count)

# Edit FENs so puzzle can be converted to playing as white
with open('training data/allPuzzles_b.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        FEN = ''
        for letter in str(row[1]):
            if letter.isupper():
                FEN = FEN + letter.lower()
            elif letter.islower():
                FEN = FEN + letter.upper()
            else:
                FEN = FEN + letter
        with open('training data/allPuzzles_conv_w.csv', 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            new_row = [row[0], FEN, row[2], row[3]]
            writer.writerow(new_row)