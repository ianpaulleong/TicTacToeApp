import numpy as np

def indicesToNum(theRow,theCol):
    thePad = np.array([[7,8,9],[4,5,6],[1,2,3]])
    return thePad[theRow][theCol]

def numToIndices(theNum):
    theRowList = [2,2,2,1,1,1,0,0,0]
    theColList = [0,1,2,0,1,2,0,1,2]
    theRow = theRowList[theNum-1]
    theCol = theColList[theNum-1]
    return np.array([theRow,theCol])

def probPicker(theArray):
    theArray = theArray[theArray != -1]
    theArrayNormalized = theArray + (1-theArray.max())
    theArrayNormalized = theArray*theArray
    theArrayNormalized /= theArrayNormalized.sum()
    theRand = np.random.rand()
    theAdd = 0
    theIndexChoice = 10^3
    for ii in range(len(theArray)):
        theAdd += theArrayNormalized[ii]
        if theAdd >= theRand:
            theIndexChoice = ii
            break
    if (theIndexChoice == 10^3) | (theIndexChoice >= len(theArray)):
        #cow = 2
        theIndexChoice = 0
    if len(theArray) == 0:
        cow = 2
    return theArray[theIndexChoice]
        
    