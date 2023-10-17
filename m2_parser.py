import csv
import itertools

# Parse mateIn2 puzzles from lichess database
count = 0
with open('chess_puzzles.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in itertools.islice(reader, 3000000):
        if str('mateIn2') in str(row) and len(str(row[2])) == 19:  # and ignore puzzles with queening
            count = count + 1
            with open('mateIn2.csv', 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(row[:4])
    print(count)

# Edit FEN to ensure puzzle is always played as white
with open('mateIn2.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if str(' w ') in str(row[1]):  ## w means puzzle is black to play
            FEN = ''
            for letter in str(row[1]):
                if letter.isupper():
                    FEN = FEN + letter.lower()
                elif letter.islower():
                    FEN = FEN + letter.upper()
                else:
                    FEN = FEN + letter
            with open('mateIn2_white.csv', 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                new_row = [row[0], FEN, row[2], row[3]]
                writer.writerow(new_row)
        else: 
            with open('mateIn2_white.csv', 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(row)