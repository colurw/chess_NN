""" Creates training, validation, and testing datasets containing one-hot tensors.  Initialises a 
neural network, trains it, assesses its ability to find the best move, then saves it.  Draws a graph 
of the training history in order to diagnose over-training or an under-powered network """

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import pickle
import random
import time
from _0_chess_tools import reshape_output, one_hot_to_unicode

# Get pickled data from encoder.py
with open("training data/all_puzz_w.pickle", "rb") as file:
    all_puzzles_1 = pickle.load(file) 
with open("training data/all_puzz_conv_w.pickle", "rb") as file:
    all_puzzles_2 = pickle.load(file) 

with open("training data/all_solns_w.pickle", "rb") as file:
    solutions_1 = pickle.load(file) 
with open("training data/all_solns_conv_w.pickle", "rb") as file:
    solutions_2 = pickle.load(file) 

# Create datasets
x_train_1 = np.array(all_puzzles_1[:1200000], dtype='bool')
x_train_2 = np.array(all_puzzles_2[:1200000], dtype='bool')
x_train = np.concatenate((x_train_1, x_train_2), axis=0, dtype='bool')

y_train_1 = np.array(solutions_1[:1200000], dtype='bool')
y_train_2 = np.array(solutions_2[:1200000], dtype='bool')
y_train = np.concatenate((y_train_1, y_train_2), axis=0, dtype='bool')

x_test_1 = np.array(all_puzzles_1[1400000:], dtype='bool')
x_test_2 = np.array(all_puzzles_2[1400000:], dtype='bool')
x_test = np.concatenate((x_test_1, x_test_2), axis=0, dtype='bool')

y_test_1 = np.array(solutions_1[1400000:], dtype='bool')
y_test_2 = np.array(solutions_2[1400000:], dtype='bool')
y_test = np.concatenate((y_test_1, y_test_2), axis=0, dtype='bool')

print('train_x:', x_train.shape, ' train_y:', y_train.shape)
print(' test_x:', x_test.shape,'   test_y:', y_test.shape)

# Random number generator
rng = str(time.time())[:-3:-1]
random.seed(rng)

# Define neural network
input = layers.Input((64,13))
flatten = layers.Flatten()(input)
noise1 = layers.GaussianNoise(stddev=0.000)(flatten)       ## 0.025, seed=7
dropout1 = layers.Dropout(0.00)(noise1)                    ## 0.05
hidden1 = layers.Dense(2048, activation='relu')(dropout1)
bnorm = layers.BatchNormalization(axis=-1)(hidden1)         ## axis=-1?
# noise2 = layers.GaussianNoise(stddev=0.000, seed=7)(bnorm)
# dropout2 = layers.Dropout(0.00)(noise2)
hidden2 = layers.Dense(1024, activation='relu')(bnorm) 
outputs = [layers.Dense(13, activation='softmax')(hidden2) for _ in range(64)]

model = keras.models.Model(inputs=input, outputs=outputs)
model.summary()

# create diagram of model
keras.utils.plot_model(model, "current model.png", show_shapes=True)

#compile model
model.compile(optimizer='adam', 
              metrics='categorical_accuracy', 
              loss=['categorical_crossentropy' for _ in range(64)])     

# Select indexes of random puzzles from train dataset without replacement
VAL_SPLIT = 0.25
rand_nums = random.sample(range(len(x_train)), int(len(x_train)*VAL_SPLIT))    

# Move selected puzzles to the end of training datasets for automatic validation_split
rand_nums.sort(reverse=True)
x_train_shuffled = np.append(x_train, x_train[rand_nums], axis=0)
x_train_shuffled = np.delete(x_train_shuffled, [rand_nums], axis=0)

y_train_shuffled = np.append(y_train, y_train[rand_nums], axis=0)
y_train_shuffled = np.delete(y_train_shuffled, [rand_nums], axis=0)

# Reshape y_data to suit network with 64 branched outputs: list[square, ndarray[puzzle, piece_ohe_vector]]
yy_train_shuffled = reshape_output(y_train_shuffled, 64)
yy_test = reshape_output(y_test, 64)

# Early stopping critera
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1, restore_best_weights=True)

# Train model
TITLE = ('n=2048,bn,1024_bs=1024_2400k_4')
start = time.time()
history = model.fit(x_train_shuffled, yy_train_shuffled, batch_size=1024, epochs=25, validation_split=VAL_SPLIT, callbacks=[es], verbose=2) 
end = time.time()
minutes = int((end-start)/60)
model.save('models/general_solver_4')

# Test model
test_scores = model.evaluate(x_test, yy_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test cat. accuracy:", 1-test_scores[1])

# Calculate overall accuracy on previously unseen test puzzles
count = 0
for i in (range(10000)):
    # Get a random unseen puzzle 
    rand_num = random.sample(range(len(x_test)), 1)
    x_sample = x_test[rand_num]
    y_truth = y_test[rand_num]
    # Evaluate board
    y_predict = model(x_sample)
    y_predict = np.array(y_predict)
    # Convert y_predict categorical probabilities and y_truth one-hot array into strings of category labels
    predict_str = one_hot_to_unicode(y_predict)
    puzzle_str = one_hot_to_unicode(y_truth)
    if predict_str == puzzle_str:
        count += 1
    score = str(round(100*count/10000, 3))
print(i, 'puzzles checked')
print(score, '% accurately solved')

# Plot training history
fig, (acc, los) = plt.subplots(1,2)
fig.suptitle('gen_solv: '+TITLE+'  '+str(minutes)+'_min'+'  '+score+'%_solved')
acc.plot(history.history['dense_21_categorical_accuracy'])
acc.plot(history.history['val_dense_21_categorical_accuracy'])
los.plot(history.history['loss'])
los.plot(history.history['val_loss'])
los.yaxis.tick_right()
acc.set_xlabel('epoch')
acc.set_ylabel('categorical accuracy')
los.set_xlabel('epoch')
los.set_ylabel('loss')
acc.grid()
los.grid()
acc.legend(['output_21 train', 'output_21 val'], loc='lower right')
los.legend(['train', 'val'], loc='upper right')
fig.text(0.12, 0.92, 'test cat. acc. (total): {}'.format(round(1-test_scores[1],4)), fontsize=9, verticalalignment='top')
fig.text(0.55, 0.92, 'test loss: {}'.format(round(test_scores[0],3)), fontsize=9, verticalalignment='top')
plt.savefig(r'training graphs/general/{}.png'.format(TITLE))
