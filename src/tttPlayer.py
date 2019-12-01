# This class is a tic-tac-toe player. It takes as an input a 3x3 numpy 
# tic-tac-toe board state and outputs a 3x3 numpy evaluation of that board 
# state. Specifically, for each spot on the board, it evaluates how good it 
# thinks it would be to make the next move at that spot. Based on that 
# evaluation it also recommends a move to make. This can be done either by 
# choosing the highest evaluated (legal) move or in a probabalistic manner
#
# Note that it doesn't know if it's playing 'X' or 'O' -- it expects that 
# self-taken spots are marked with a '1', and opponent-taken spots are marked 
# with a '-1', with '0' # indicating an empty spot.
#
# Evaluations have a range from 0 to 1 for each of the nine spots on a 
# tic-tac-toe board, where 0 indicates a losing move and 1 a winning move. 
#
# This tttPlayer class has several modes of operation: random, network, and 
# optimal. 
# Random: randomly generate an evaluation of the board
# Network: use a neural network to evaluate the board
# Optimal: use a lookup table to evaluate the board. This lookup table was 
#          created by brute-forcing all possible games of tic-tac-toe. In this 
#          mode, the tttPlayer will never lose.
# 

# import various libraries
import numpy as np
import random as rand
import torch
import pickle

# import some useful utility functions
from utilityFunctions import indicesToNum
from utilityFunctions import numToIndices
from utilityFunctions import probPicker

# Define the class!
class tttPlayer:
    
    
    # Class creator!
    # Note: the default mode is random. 
    # Note: if we're in 'network' mode, the second input will need to be a 
    #       neural network. Since we're using pyTorch, this network will take a
    #       9x1 tensor input of board state and spit out a 9x1 tensor output of
    #       the network's evaluation.
    # Note: the Probability-Based-Picker referenced by probPickerOn is set to 
    #       '1' (for 'on'), the move chosen by the tttPlayer is not necessarily
    #       the highest-scoring move. Instead, the tttPlayer picks moves 
    #       probabalistically, with higher-scoring moves being more likely.
    # Note: there is an input for network type. Currently Convolutional Neural 
    #       Networks have not been implemented
    def __init__(self, whatMode = "random", theNet = 0, probPickerOn = 1, netType = "nn"):
        # Set the mode of the picker
        self.mode = whatMode
        if self.mode == "network":
            self.net = theNet
            self.netType = netType
            self.probPickerOn = probPickerOn
        elif self.mode == "optimal":
            # This is a liiiiiiitle kludgey; ideally, I should give the 
            # location of the table as an input to the __init__ function. 
             with open('optimalTable.pickle', 'rb') as handle:
                 cow = pickle.load(handle)
             self.optimalTable = cow
            
    # This function takes a numpy 3x3 board state and spits out what move to 
    # make. The moves are recorded as a number from 1 to 9, corresponding to
    # a keyboard number pad. Like so:
    # 7|8|9
    # 4|5|6
    # 1|2|3
    def chooseMove(self,board):
        # This function takes a 3x3 numpy board state and outputs a chosen move
        
        if self.mode == "random":
            # find where the board has empty spaces and arbitrarily pick one.
            whereEmpty = board == 0
            numEmpty = sum(sum(whereEmpty))
            indicesEmpty = np.where(whereEmpty)
            whichMove = rand.randint(0,numEmpty-1)
            whereMoveRow = indicesEmpty[0][whichMove]
            whereMoveCol = indicesEmpty[1][whichMove]
            return indicesToNum(whereMoveRow,whereMoveCol)
            
        
        elif self.mode == "optimal":
            # Okay, this is SUPER kludgey. See, I made the 'random' and 
            # 'network' modes first, and only after I'd trained a really good 
            # neural network did I bother to make the optimal mode. By this 
            # point, I'd forgotted that tttPlayer takes board state as input 
            # and not move history, BECAUSE I STUPIDLY HADN'T COMMENTED MY 
            # CODE. So when making a brute-force table for the 'optimal', I 
            # built the table such that it took move history as an input, and 
            # not board state.
            # Anyhow, this entire first section is me taking a board state and
            # creating a (fake) move history out of it. Fortunately, the 
            # evaluation of a tic-tac-toe board state doesn't depend on how we
            # got to the board state, so any move-history that leads to the 
            # current board state will get me a useful evaluation out of the 
            # table
            boardStateIndex = 0
            if sum(sum(board)) <= 0:
                board *= -1
            numX = 0
            numO = 0
            for jj in range(1,10):
                theRowCol = numToIndices(jj)
                theRow = theRowCol[0]
                theCol = theRowCol[1]
                if board[theRow,theCol] == 1:
                    numX += 1
                    boardStateIndex += jj*(10**(2*numX-2))
                elif board[theRow,theCol] == -1:
                    numO += 1
                    boardStateIndex += jj*(10**(2*numO-1))
            # You are now leaving Kludge Central. Temporarily. 
            
            # Pick the best move. If multiple moves are tied for best (i.e. 
            # multiple winning moves, or if none, multiple tying moves), pick 
            # one arbitrarily.
            theOptimalTable = self.optimalTable[boardStateIndex]
            theMax = theOptimalTable.max()
            optMoveList = []
            for jj in range(1,10):
                if theOptimalTable[jj] == theMax:
                    optMoveList.append(jj)
            return rand.choice(optMoveList)
            
            
        elif self.mode == "network":
            if self.netType == "nn":
                # Call the 'Evaluate Board' function to get a 3x3 numpy set of 
                # scores that indicate, for each position on the board, how 
                # good making the next move at that position is.
                theEval = self.evaluateBoard(board)
                # Set the score of positions that have been taken already to an
                # arbitrarily low value
                whereTaken = board != 0
                theEval[whereTaken] = -1
                # Choose where to move. If the Probability-Based Picker is 
                # active, this may not be the highest-evaluated location. Note
                # that what the probPicker spits out is the evaluation at the 
                # chosen location, not the location itself. I had good reasons
                # to do it this way.
                # Really, I did.
                # No, not really.
                if self.probPickerOn:
                    theChoice = probPicker(theEval)
                else:
                    theChoice = theEval.max()
                    
                # This is where I actually find the chosen location.
                indicesOfChoice = np.where(theEval == theChoice)
                if len(indicesOfChoice[0]) != 1:
                    # I ran into an issue where sometimes two spots had the 
                    # same evaluation, usually because they both immediately 
                    # won. Look, just pick one, okay?
                    theRand = np.random.randint(len(indicesOfChoice[0]))
                    theRow = int(indicesOfChoice[0][theRand])
                    theCol = int(indicesOfChoice[1][theRand])
                else:
                    theRow = int(indicesOfChoice[0])
                    theCol = int(indicesOfChoice[1])
                
                return(indicesToNum(theRow, theCol))
            else:
                # I don't know if this section will ever be used; it exists in 
                # case I decide to add CNN capability
                cow = 2
            
    
    def evaluateBoard(self,board):
        # This function takes a 3x3 numpy board state and outputs a 3x3 numpy 
        # set of scores that indicate, for each position on the board, how good
        # making the next move at that position is.
        
        if self.mode == "random":
            # Do I really need to explain this? 
            # If the mode is random
            # Give a random output
            return np.random.random((3,3))/5
            
        elif self.mode == "optimal":
            # Um. Yeah, go up to the 'optimal' section of the 'chooseMove' 
            # function, where I explain this Kludge. Suffice to say, I'm 
            # converting board state into a fake move list because reasons.
            boardStateIndex = 0
            if sum(sum(board)) != 0:
                board *= -1
            numX = 0
            numO = 0
            for jj in range(1,10):
                theRowCol = numToIndices(jj)
                theRow = theRowCol[0]
                theCol = theRowCol[1]
                if board[theRow,theCol] == 1:
                    numX += 1
                    boardStateIndex += jj*(10**(2*numX-2))
                elif board[theRow,theCol] == -1:
                    numO += 1
                    boardStateIndex += jj*(10**(2*numO-1))
            
            # Ugh, why did I have to do such a lazy job with the brute force 
            # table? Using the Kludged fake move list, get an evaluation in the
            # wrong shape and reshape it to produce an output. There's a better
            # way to do this than this, but whatever.
            theOptimalTable = self.optimalTable[boardStateIndex]
            theOutTable = np.zeros([3,3])
            for jj in range(1,10):
                theRowCol = numToIndices(jj)
                theRow = theRowCol[0]
                theCol = theRowCol[1]
                theOutTable[theRow,theCol] = theOptimalTable[jj]
            
            return theOutTable
            
        
        
        elif self.mode == "network":
            # Yay, this bit's easy! Take the input, make it into a tensor, 
            # throw it into the neural network, make the output into a 3x3 
            # numpy thing and spit it back out.
            if self.netType == "nn":
                evalTens = self.net(torch.tensor(board.reshape(9), dtype = torch.float32))
                return evalTens.detach().numpy().reshape([3,3])
            else:
                cow = 2
        