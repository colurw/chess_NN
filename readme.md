# Chess_NN

Chess_NN is chess engine powered by an ensemble of neural networks.  It has a web-app 
frontend, playable in a browser, by virtue of Python's Django web framework.  When 
given a board state, it attempts to predict what the best move would be, as according 
to the Stockfish chess engine, and then plays that move.

Stockfish is the dominant chess engine, capable of beating _the_ best human player
98% of the time.  It does this by searching through the exponentially-expanding tree
of possible moves, for as many future turns as it can.  Typically users try to reduce
its abilities by limiting the time allowed for this search, or making it occasionally
play non-optimum moves.

When tested on previously unseen board-states from the testing dataset, Chess_NN 
currently predicts Stockfish's optimum move for 47% of boards, and does so roughly 
1000x faster.  However, it is also unable to predict any move 24% of the time, so 
resorts to making a random legal move. In the remaining 19% of cases, it plays a 
legal move of indeterminate quality.

## How A Neural Network Learns Chess

"Everything in chess is pattern recognition" - _I.M. Robert Ris_

The training data for the neural network is taken from Lichess.com's puzzles database,
which contains 1.2 million mid-game and end-game board states taken from human games,
with the 'correct answer' being assessed by Stockfish.  The training data does not 
contain any openings (the first 10 or so moves by each player), hence the user is
prompted to choose from a selection of fully-developed openings that give both players 
a fair chance of winning.

The neural network is based on the Keras Functional API, which allows the user to 
buildmore complex networks with non-standard topologies.  The ChessNN input layer 
accepts a 64x13 one-hot array (64 squares than can be filled with one of 6 black, 
6 white, or no pieces) and the output is 64 multi-class classifiers.  Each classifer
is represented by a soft-max layer that receives a single 13-feature probability vector.
  
Testing showed that wide and shallow networks performed better (and trained faster) 
than deeper networks with a similar number of neurons, and that injecting a low level 
of noise into the input layer improved it ability to generalise, at the expense of 
training time.

Experiments with adding noise during solving, then taking the average of all solutions
led to training several models on a portion of the training data each.  By combining
the resulting predictions in various ways we can avoid many instances of the two most
common failure modes: a piece being cloned during a move, or a piece disappearing
during a move.

## Is Chess_NN actually any good at playing chess?

Not really.  In restricted/idealised test conditions 24% of its moves are random picks 
from a list of legal moves generated by the Python-Chess 
library.  This occurs when all the stages of the ensemble's voting criteria have 
been exhausted without producing a sensible result.  In real gameplay, this means
the model can play several random moves in a row, which can easily be taken 
advantage of by an average human player, leading to an unsurmountable upper hand.

However... It is possible to extract much more (5-8x) training data from 
the Lichess.com puzzle sequences (I only used the first step of each one).  And, if 
more RAM was made available much larger networks could be trained. Making both these 
improvements would almost certainly result in a significant boost in performance.

## What's Under The Hood

### _0_chess_tools.py  
Contains functions shared by the other python files, when 
converting between various data formats, checking for illegal moves, displaying the
game state, or comparing predicted moves to Stockfish ground truth moves.

### _1_parser.py  
Reads chess puzzles written in Forsyth-Edwards notation from 
Lichess.com's puzzle database.  Data is cleaned and parsed according to puzzle-type.  
Black-to-play games converted into white-to-play for consistency during training.

### _2_encoder.py  
Takes the parsed FENs and converts them into one-hot tensors.  The 
one-hot tensors are an array with 64 rows (each represents a square on the chess 
board) and 13 columns (each represents a possible piece that can occupy a square).  
These are a sparse data format, containing mostly zeroes. 
Applying the first move from the Lichess data creates the x_data, then applying 
the second move creates the y_data

### _3_data_viewer.py  
When given a one-hot tensor, creates .png image of the chess 
board.  Used to check the training data is as expected / free of errors.

### _4_trainer.py  
Creates training, validation, and testing datasets containing x,y pairs of one-
hot tensors that represent puzzle board states.  Initialises a neural network, trains it, 
assesses its ability to findthe best move, and saves the model.  Plots the training 
history in order to help diagnose over-training or an under-powered network.

### _5_solver.py  
Picks random puzzles from the testing dataset, then compares the 
neural network's prediction of the solution against the ground truth solution from 
the database.  Four _puzzle-solution-prediction_ triplets are converted into a graphic 
and saved as a .png file.  These can be found in the /results/ folder.

![image](https://github.com/colurw/chess_NN/assets/66322644/6fe6e368-3d26-414a-b4c0-a2235b706bf2)

### _6_solver_ensemble.py  
Several neural networks are presented with the same board state.  The 
predicted solutions are combined according to decision criteria to produce an 
output that is substantially more accurate than any single neural network is capable 
of producing, analogous to classic wisdom-of-the-crowds quantity estimation findings.

### _7_ensemble_graph_maker.py  
Visualises the performance gains made by increasing the
number of neural networks in the ensemble.  

General Solver - Ensemble Results | Mate In One - Ensemble Results
:-------------------------:|:-------------------------:
![image](https://github.com/colurw/chess_NN/assets/66322644/a8d73813-49a5-42b6-93f0-e1c9a1715ff1)  |  ![image](https://github.com/colurw/chess_NN/assets/66322644/0d6b8c08-e593-4aaa-992f-24eb778094cf)

### django web framework
<img src="https://github.com/colurw/chess_NN/assets/66322644/b3d419ff-06b9-4444-85ba-99531d4db79c" align="right" width="350px"/>
Creates an IP connection to the browser over the Localhost.  When Views.py is 
called by Urls.py it returns data that populates the Index.html template with the 
current board image and relevant messages.  Form data from the browser are sent back
to views.py as POST get requests, converted into tensors, then passed to 
Ensemble_solver(), which returns a tensor representing the move to be played in 
response.  This tensor is converted by Local_chess_tools.py into an image of the next 
board state, and then into a 64-bit string, which can be sent as an argument of 
HttpRequest() back to Index.html
As the training data does not include early-game board states, the user must initially 
select from one of three fully-developed opening options.  This also avoids having to 
implement a castling feature - moves of which were also excluded from the training 
dataset to allow less-complex functions when encoding raw training data.
<br clear="right"/>

### m2_[filename].py  
Does the same things as the above files, but only considers a subset of
the Lichess database - the 'Mate In Two' puzzles.  Chess_NN is surprisingly good at 
solving these, considering it has to predict a sequence of three moves correctly - an 
amount of moves that prevents any reasonably-implementable error/illegal move 
checking routines.  This performance is likely achieved by the smaller (and less 
varied) dataset being able to be more comprehensively modelled, by a neural network 
limited to local hardware resources.

