""" Creates training, validation, and testing datasets containing one-hot tensors.  Initialises a 
neural network, trains it, assesses its ability to find the best move, then saves it.  Draws a graph 
of the training history in order to diagnose over-training or an under-powered network """

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import random
import time
from _0_chess_tools import reshape_output, one_hot_to_unicode
import gc
import os

MODEL_NAME = 'whole_game_4'
LOAD_PRETRAINED = False
RANDOM_SEED = 7
DATA_LIMIT = 4000000
TEST_SPLIT = 0.05
VAL_SPLIT = 0.1
MEMORY_MAPPING = True
EPOCHS = 40     
PATIENCE = 3
BATCH_SIZE = 1024
EVALUATE_BOARDWISE_ACCURACY = False
DRAW_GRAPH = False
NOTES = f'n=1500,bn,1500_bs={BATCH_SIZE}_{str(DATA_LIMIT)[0:4]}k_wg4'

# Get data from pgn_parser_encoder.py
x_data = np.load('training data/whole_game_x.npy')
y_data = np.load('training data/whole_game_y.npy')

x_data = x_data[0 : DATA_LIMIT]
y_data = y_data[0 : DATA_LIMIT]

# Shuffle and split datasets
random.seed(RANDOM_SEED)
rand_nums = random.sample(range(len(x_data)), int(len(x_data))) 

x_data_shuffled = [x_data[index] for index in rand_nums]
y_data_shuffled = [y_data[index] for index in rand_nums]

x_train = np.array(x_data_shuffled[int(len(x_data)*TEST_SPLIT) : ], dtype='bool')
y_train = np.array(y_data_shuffled[int(len(x_data)*TEST_SPLIT) : ], dtype='bool')

x_test = np.array(x_data_shuffled[0 : int(len(x_data)*TEST_SPLIT)], dtype='bool')
y_test = np.array(y_data_shuffled[0 : int(len(x_data)*TEST_SPLIT)], dtype='bool')

np.save('whole_game_x_test.npy', x_test)
np.save('whole_game_y_test.npy', y_test)

print('x_train:', x_train.shape, ' y_train:', y_train.shape)
print('x_test:', x_test.shape,'  y_test:', y_test.shape)

# Reshape y data to suit network with 64 branched outputs: list[board_square, ndarray[puzzle_id, piece_ohe_vector]]
yy_train = reshape_output(y_train, 64)
yy_test = reshape_output(y_test, 64)


if MEMORY_MAPPING:
    try:
        os.makedirs('.memmaps')
    except:
        pass

    x_train_mem = np.memmap('.memmaps/x.mmap', dtype='bool', mode='w+', shape=x_train.shape)
    x_train_mem[:] = x_train[:]

    yy_train_mem = [np.memmap(f'.memmaps/y{i}.mmap', dtype='bool', mode='w+', shape=(y_train.shape[0], 13)) for i in range(64)]
    for i, m_map in enumerate(yy_train_mem):
        m_map[:] = yy_train[i][:]

    # Test for offset errors
    assert x_train_mem.all() == x_train.all()
    assert set([yy_train_mem[i].all() == yy_train[i].all() for i in range(64)]) == {True}

    del x_train, y_train, yy_train
    del x_data, y_data
    gc.collect()

else:   
    x_train_mem = x_train 
    yy_train_mem = yy_train 


if LOAD_PRETRAINED:
    model = keras.models.load_model(f'models/{LOAD_PRETRAINED}')
else:    
    # Define neural network
    input = layers.Input((64,13))
    flatten = layers.Flatten()(input)
    noise1 = layers.GaussianNoise(stddev=0.000)(flatten)       ## 0.025
    dropout1 = layers.Dropout(0.00)(noise1)                    ## 0.05
    hidden1 = layers.Dense(1500, activation='relu')(dropout1)
    bnorm = layers.BatchNormalization(axis=-1)(hidden1)         
    hidden2 = layers.Dense(1500, activation='relu')(bnorm) 
    outputs = [layers.Dense(13, activation='softmax')(hidden2) for _ in range(64)]

    model = keras.models.Model(inputs=input, outputs=outputs)
    model.summary()

    # Compile model
    model.compile(optimizer='adam', 
                metrics='categorical_accuracy', 
                loss=['categorical_crossentropy' for _ in range(64)])     


# Early stopping critera
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                      mode='min', 
                                      patience=PATIENCE, 
                                      verbose=1, 
                                      restore_best_weights=True)
                                    
# Train model
start = time.time()
history = model.fit(x_train_mem, yy_train_mem, 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_split=VAL_SPLIT, 
                    callbacks=[es], 
                    verbose=2) 

end = time.time()
minutes = int((end-start)/60)
model.save(f'models/{MODEL_NAME}')

# Test model
test_scores = model.evaluate(x_test, yy_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test cat. accuracy:", 1-test_scores[1])


if EVALUATE_BOARDWISE_ACCURACY:
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
else:
    score = '?'


if DRAW_GRAPH:
    print(i, 'moves tested')
    print(score, '% board-states accurately predicted')

    # Plot training history
    fig, (acc, los) = plt.subplots(1,2)
    fig.suptitle(f'{MODEL_NAME}: {NOTES} {str(minutes)}_min {score}%_solved')
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
    fig.text(0.12, 0.92, f'test cat. acc. (total): {round(1-test_scores[1],4)}', fontsize=9, verticalalignment='top')
    fig.text(0.55, 0.92, f'test loss: {round(test_scores[0],3)}', fontsize=9, verticalalignment='top')
    plt.savefig(f'training graphs/whole_game/{NOTES}.png')


