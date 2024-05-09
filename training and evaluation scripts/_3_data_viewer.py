""" Choose sample one-hot tensors, and view a graphic version of the chess board.  A sense check. """

import pickle
import matplotlib.pyplot as plt
from _0_chess_tools import one_hot_to_png, im_concat_2

# Get data from chess_encoder.py
with open("training data/all_puzz_conv_w.pickle", "rb") as file:
    puzzles = pickle.load(file) 
with open("training data/all_solns_conv_w.pickle", "rb") as file:
    solutions = pickle.load(file) 

# Display puzzle and solution for given indexes
sample_list = [250, 300, 350, 400, 450, 500]
for frame in sample_list:
    img1 = one_hot_to_png(puzzles[frame])
    img2 = one_hot_to_png(solutions[frame])
    img = im_concat_2(img1, img2)
    plt.imshow(img)
    plt.show()