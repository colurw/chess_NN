""" Receives setup and player's move data posted from index.html.  Returns a http response. """

from django.shortcuts import render
from django.http import HttpResponse
from io import BytesIO
import base64
import pickle
import numpy as np
from . import chess_tools_local as ct 
from . import web_ensemble_solver as es


def check_input(move):
    """ checks whether move is of format 'a2a3' """
    valid_chars = "12345678abcdefgh"
    if type(move) == str:
        if (len(move) != 4
                or move[0].islower() == False 
                or move[1].isnumeric() == False 
                or move[2].islower() == False 
                or move[3].isnumeric() == False 
                or set(move) - set(valid_chars) != set()):
            return 'fail'
    else:
        return 'fail'


def check_input_q(move):
    """ checks whether move is of format 'a7a8q' to allow queening """
    valid_chars = "12345678abcdefghq"
    if type(move) == str:
        if (len(move) != 5
                or move[0].islower() == False 
                or move[1].isnumeric() == False 
                or move[2].islower() == False 
                or move[3] != '8' 
                or move[4] != 'q'
                or set(move) - set(valid_chars) != set()):
            return 'fail'
    else:
        return 'fail'


# Convert image to base64 string
def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())
    img_str = img_str.decode("utf-8")
    return img_str


def index(request):
    """ called by urls.py when /index.html is requested by browser, returns http response """
    move = None
    ai_move = None
    last_move = None
    valid_input = False
    fen = '8/8/8/8/8/8/8/8 w KQkq - 0 1'  

    # If opening option 1 selected, set FEN to King's Indian Defence
    if request.POST.get('option1') == 'Go':
        fen = '1KR1QB1R/PPPB2PP/2N2N2/3PPP2/3p4/1pn1p3/pbp2ppp/1kr1qbnr w KQkq - 0 1'
        move = ' ...play g1f3 to finish opening'
        with open("webapp/temp/saved_fen", "wb") as file:
            pickle.dump((fen), file)
    
    # If opening option 2 selected, set FEN to Nimzo-Indian Defence
    elif request.POST.get('option2') == 'Go':
        fen = '1KR1QB1R/PPP3PP/2NPBN2/4PPb1/4pp2/2np4/ppp3pp/1kr1qbnr w KQkq - 0 1'
        move = ' ...play g1f3 to finish opening'
        with open("webapp/temp/saved_fen", "wb") as file:
            pickle.dump((fen), file)
    
    # If opening option 1 selected, set FEN to Ruy Lopez
    elif request.POST.get('option3') == 'Go':
        fen = '1K1RQBNR/PPP1P1PP/2N2PB1/3P4/3p2p1/2n2n1p/pppbpp2/1kr1qb1r w KQkq - 0 1'
        move = ' ...play e2e4 to finish opening'
        with open("webapp/temp/saved_fen", "wb") as file:
            pickle.dump((fen), file)

    # If player move posted, record move
    elif request.method == "POST":
        move = request.POST.get('human_move')
        
        # Check input string from index.html is valid, eg: 'b3c4' or 'd7d8q'
        if check_input(move) == 'fail' and check_input_q(move) == 'fail':
            
            return HttpResponse("invalid input, use format: 'a2a3', or 'a7a8q' if queening a pawn")
              
        else:
            valid_input = True 
            # Check move is legal according to chess rules
            with open("webapp/temp/saved_fen", "rb") as file:
                fen = pickle.load(file)
            moves = [str(move[:2]), str(move[2:])]
            # ..from black's perspective
            flipped_fen = ct.swap_fen_colours(fen, turn='white') 
            if ct.is_move_legal(flipped_fen, moves) == False:
                
                return HttpResponse("illegal move detected!")

        # Load FEN
        with open("webapp/temp/saved_fen", "rb") as file:
            fen = pickle.load(file) 
        # Convert FEN to one-hot tensor and apply human move
        board = ct.fen_to_ascii(fen)
        onehot = ct.one_hot_encode(board)
        if valid_input == True:                
            onehot = ct.update_one_hot(onehot, move)
        # Get ensemble prediction of best computer move
        onehot = np.array(onehot).reshape(1,64,13)
        onehot, ai_move, tag = es.ensemble_solver(onehot)
        # Save updated FEN
        fen = ct.one_hot_to_fen(onehot)
        with open("webapp/temp/saved_fen", "wb") as file:
            pickle.dump((fen), file)
        # Convert onehot tensor to board image then convert image to string
        image = ct.one_hot_to_png(onehot)
        image64 = image_to_base64(image)
        
        return render(request, "index.html", {'ai_move': ai_move, 'move': move, 'image64': image64, 'fen': fen, 'tag': tag})

    # Convert seleceted opening FEN to one-hot tensor
    board = ct.fen_to_ascii(fen)
    onehot = ct.one_hot_encode(board)        
    # Draw image and save as string
    image = ct.one_hot_to_png(onehot)
    image64 = image_to_base64(image)

    return render(request, "index.html", {'ai_move': ai_move, 'move': move, 'image64': image64})