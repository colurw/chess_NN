""" Picks random puzzles from the testing dataset, and compares the neural network's prediction of 
the solution against the actual solution from the database.  Four puzzle-solution-prediction triplets 
are converted to a graphic and saved as a .png file """

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from _0_chess_tools import one_hot_to_png, one_hot_to_unicode, im_concat_4, im_concat_3, is_only_one_move
import time

# Get data from encoder.py
print('loading pickled data...')
with open("training data/mate_in_1.pickle", "rb") as file:
    mate_in_one = pickle.load(file) 
with open("training data/solution.pickle", "rb") as file:
    solution = pickle.load(file) 

# Create datasets
print('\ncreating test dataset...')
x_test = np.array(mate_in_one[1300000:], dtype='bool')
y_test = np.array(solution[1300000:], dtype='bool')
print(x_test.shape, y_test.shape, '\n')

# Load model from chess_trainer.py
model = keras.models.load_model('models/matein1b')

# calculate overall accuracy
count = 0
for i in (range(500)):
    # Get random board
    rand_num = random.sample(range(len(x_test)), 1)
    x_sample = x_test[rand_num]
    y_truth = y_test[rand_num]
    # Evaluate board
    y_predict = model(x_sample)
    y_predict = np.array(y_predict)
    # Convert prediction categorical probabilities and truth one-hot array into strings of category labels
    pred_str = one_hot_to_unicode(y_predict)
    puzzle_str = one_hot_to_unicode(y_truth)
    # Compare strings
    if pred_str == puzzle_str:
        count += 1
    score = str(round(100*count/500, 2))
print(score, '% of puzzles accurately solved')

for num in range(10):
    # Solve four puzzles from the test dataset
    images = []
    for i in (range(4)):
        rand_num = random.sample(range(len(x_test)), 1)
        x_sample = x_test[rand_num]
        y_truth = y_test[rand_num]
        # Evaluate board
        y_predict = model(x_sample)
        y_predict = np.array(y_predict)
        # Convert output and stitch images
        im = im_concat_3(one_hot_to_png(x_sample), 
                    one_hot_to_png(y_truth), 
                    one_hot_to_png(y_predict))
        images.append(im)
        #print(is_only_one_move(x_sample, y_predict))

    # Stitch all solved puzzles
    timestr = time.strftime("%Y%m%d-%H%M%S")
    im_concat_4((images[0]), 
            (images[1]),
            (images[2]), 
            (images[3])).save('results/M2_{}.png'.format(timestr))

    # Display image
    img = mpimg.imread('results/M2_{}.png'.format(timestr))
    plt.imshow(img)
    plt.show()