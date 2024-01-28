# Chess_NN

Chess_NN is chess engine powered by an ensemble of neural networks.  It has a web-app 
frontend, playable in a browser, by virtue of Python's Django web framework.  When 
given a board state, it attempts to predict what the best move would be - according to 
Stockfish chess engine, and then plays that move.

Stockfish is the dominant chess engine, capable of beating the current best human player
98-99% of the time.  It does this by searching through an exponentially-expanding tree
of possible moves, for as many future turns as time allows.  Typically competitors try to 
reduce its performance by limiting the duration of this search, or by forcing it to select
less-optimal moves.

When tested on previously unseen chess puzzles from the testing dataset, Chess_NN 
currently predicts Stockfish's optimum move for 47% of boards, and does so roughly 
1000x faster.  However, it is unable to predict any move 24% of the time, so 
resorts to making a randomly-chosen legal move. In the remaining 29% of cases, it 
plays a non-optimal legal move of indeterminate quality.

## How A Neural Network Learns Chess

"Everything in chess is pattern recognition" - _I.M. Robert Ris_

The training data for the neural network are taken from Lichess.com's puzzles database,
which contains 1.2 million mid-game and end-game board states taken from human games,
with the 'correct answer' being assessed by Stockfish.  The training data do not 
contain any openings (the first 10 or so moves by each player), hence the user is
prompted to choose from a selection of fully-developed openings.  This also handily avoids
having to implement a castling move function. 

The neural network is based on the Keras Functional API, which allows the user to 
build networks with non-standard topologies.  The ChessNN input layer 
accepts a 64x13 one-hot array (64 squares than can be empty, or filled by one of 6 black, 
or 6 white, pieces).  The outputs are 64 multi-class classifiers.  Each classifier
is a softmax layer that produces a 13-feature probability vector - where the highest 
value identifies the predicted piece (or empty space) in that square.  

![image](https://github.com/colurw/chess_NN/assets/66322644/d7e9fce8-5cc0-487e-88b7-6c2e12d32a28 "chess_NN network diagram")

Testing showed that wide and shallow networks performed better (and trained faster) 
than deeper networks with a similar number of neurons, and that injecting a low level 
of noise into the input layer improved its ability to generalise, at the expense of 
training time.

Experimenting with adding noise during solving, then taking the average of many solutions,
showed the benefits ensemble predictions.  By training several models, each on a portion of the training data, we can combine
the resulting predictions in various ways.  This helps to avoid many instances of the two most
common failure modes: a piece being cloned, or a piece disappearing, during a move.

## Is Chess_NN any good at playing chess?

Not really...  In restricted/idealised test conditions 24% of its moves are random 
picks from a list of legal moves generated by the Python-Chess library.  This occurs 
when all the stages of the ensemble's voting criteria have been exhausted without 
producing a sensible result.  In real gameplay, this means that the model can play a 
streak of poor random moves in a row, which can be taken advantage of by a 
reasonable human player, leading to an unsurmountable upper hand.

However, it may be possible to extract much more (4-8x) training data from 
the Lichess.com puzzle sequences (The encoder currently only uses the first step of each sequence).
Given extra RAM, larger neural networks could be trained. Making both
of these improvements would almost certainly result in a boost in performance.

## What's Under The Hood

### _0_chess_tools.py  
Contains functions used by the other python files, when 
converting between various data formats, checking for illegal moves, displaying the
game state, or comparing predicted moves to Stockfish ground truth moves.

### _1_parser.py  
Reads chess puzzles written in Forsyth-Edwards Notation (FEN) from 
Lichess.com's puzzle database.  Data are cleaned and parsed according to puzzle-type.  
Black-to-play games converted into white-to-play for consistency during training.

### _2_encoder.py  
Takes the parsed FENs and converts them into one-hot tensors.  The 
one-hot tensors are an array with 64 rows (each represents a square on the chess 
board) and 13 columns (each represents a possible piece that can occupy a square).  
These are sparse data, containing mostly zeroes. 
Applying the first move from the Lichess data creates the x_data, then applying 
the second move creates the y_data

### _3_data_viewer.py  
Renders a one-hot tensor into a .png image of a chess 
board. The pieces are ASCII charcters typed over the squares whilst iterating through the one-hot tensor.
Used to check the training data are as expected / free of errors.

### _4_trainer.py  
<img src="https://github.com/colurw/chess_NN/assets/66322644/d768ea9f-b188-4423-8232-8ba7712a8182" align="right" width="450px"/>
Creates training, validation, and testing datasets containing x,y pairs of one-
hot tensors that represent puzzle board states.  A neural network is initialised, trained, 
assessed for its ability to find the optimal move, then saved.  The training 
history is plotted, in order to help diagnose over-training or an under-powered network. <br><br>
The graph (right) shows an over-fitted neural network.  After the eighth epoch of
training, the validation loss starts to increase, whilst the training loss continues to
decrease.  This happens when the model stops learning and instead begins to memorise the 
training data.  This leads to some loss of its ability to generalise, when solving 
previously-unseen puzzles. <br><br>
The accuracy graph (left) is less useful here due to the small sample size - the value is 
calculated based on one the results from a single board square. <br clear="right"/>

### _5_solver.py  
Picks random puzzles from the testing dataset, then compares the 
neural network's prediction of the solution against the ground truth solution from 
the database. _Puzzle-solution-prediction_ triplets are converted into a graphic 
and saved as a .png file.  These can be found in the /results/ folder.

Checkmate-In-One Puzzle -> Stockfish Calculated Solution -> Chess_NN Predicted Solution:
![image](https://github.com/colurw/chess_NN/assets/66322644/6fe6e368-3d26-414a-b4c0-a2235b706bf2)

### _6_solver_ensemble.py  
Several neural networks are presented with the same board state.  The 
predicted solutions are combined according to decision criteria to produce an 
output that is substantially more accurate than any single neural network is capable 
of producing, analogous to classic wisdom-of-the-crowds quantity estimation findings.

The decision critera check the legality of the raw average of the predicted moves, before
excluding illegal moves or predictions with disappearing/cloned pieces.  If the average still
does not qualify as a legal move, the most confident prediction is chosen, based on an analysis
of distances in the predicted piece vectors. If no legal predictions are found, a random move is
chosen from a list of all legal moves.

### _7_ensemble_graph_maker.py  
Visualises the performance gains made by increasing the
number of neural networks in the ensemble.  
<img src="https://github.com/colurw/chess_NN/assets/66322644/08efebbb-eae9-40ed-8d75-311a526c9108" align="left" width="375px"/> <img src="https://github.com/colurw/chess_NN/assets/66322644/c0037b89-711b-40ab-9581-9126eed443f0" align="left" width="375px"/>
<br clear="left"/>

### django web framework
<img src="https://github.com/colurw/chess_NN/assets/66322644/b3d419ff-06b9-4444-85ba-99531d4db79c" align="right" width="300px"/>
Creates an IP connection to the browser over the Localhost.  When Views.py is 
called by Urls.py, it returns data that populate the Index.html template with the 
current board image and relevant messages.  <br><br>
Form data from the browser are sent back to views.py as POST requests, converted
into tensors, then passed to Ensemble_solver(), which returns a tensor representing 
the move to be played in response.  <br><br>
This tensor is converted by Local_chess_tools.py into an image of the next 
board state, and then into a base64 string, which can be sent as an argument of 
HttpRequest() back to Index.html <br><br>
As the training data do not include early-game board states, the user must initially 
select from one of three fully-developed opening options.  This avoids having to 
implement a castling feature - moves of which were also excluded from the training 
dataset to allow less-complex functions when encoding raw training data. <br clear="right"/>

### m2_[filename].py  
Does the same things as the above files, but only considers a subset of
the Lichess database - the 'Mate In Two' puzzles.  Chess_NN is surprisingly good* at 
solving these, considering it has to predict a sequence of three moves correctly - an 
amount of moves that prevents any reasonably-implementable error/illegal move 
checking routines.  This performance is likely achieved by the smaller (and less 
varied) dataset being able to be more comprehensively modelled by a neural network 
limited to local hardware resources.

*All 64 squares correctly predicted on 58% of test_set boards.

Mate-In-Two Puzzle -> Stockfish Calculated Solution -> Chess_NN Predicted Solution:
![image](https://github.com/colurw/chess_NN/assets/66322644/871d2196-3fe9-4277-abbf-877b3dca8826)


