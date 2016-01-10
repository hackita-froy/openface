import os
import numpy as np
import matplotlib.pyplot as plt


def makeReportImg(queryPath, queryFname,
                  matchesPath, matchesFnames,
                  labels=None):

    assert type(queryPath)==str and type(queryFname)==str and type(matchesPath)==str, \
        "queryPath, queryFname, matchesPath must be strings"

    assert type(matchesFnames)==list and all([type(q)==str for q in matchesFnames]), \
        "matchesFnames must be a list of strings"

    K = len(matchesFnames)

    gotLabels = False
    if not (labels is None):
        gotLabels = True
        msg="labels must be a list / numpy 1d array of bools, of the same length as matchesFnames"
        if type(labels)==list:
            assert len(labels)==K and all([type(l)==bool for l in labels]), msg
        elif type(labels)==np.ndarray:
            assert len(labels.shape) == 1, msg
            assert len(labels)==K and labels.dtype == np.dtype(bool), msg
        else:
            print msg
            raise

    fnames = []
    fnames[0] = os.path.join(queryPath, queryFname)
    for m in matchesFnames:
        fnames.append(os.path.join(matchesPath, m))

    I = []
    for f in fnames:
        assert os.path.isfile(f), "cannot find file: " + f
        I.append(plt.imread(f))

    allRGB = all([len(img.shape)==3 and img.shape[2]==3 for img in I]) and \
             all([img.dtype==np.dtype(np.uint8) for img in I])

    assert allRGB, "eehhmmm... not images read are rgb... fix this plz..."

    nRows = max([img.shape[0] for img in I])
    nCols = sum([img.shape[1] for img in I])

    IMG = np.ndarray((nRows,nCols,3),dtype = np.uint8)
    IMG[...]=0

    iCol = 0

    for iImg, img in enumerate(I):
        if gotLabels:
            if iImg == 0:
                colorImg(img,'b')
            elif labels[iImg-1] == True:
                colorImg(img,'g')
            elif labels[iImg-1] == False:
                colorImg(img,'r')

        curCols = img.shape[0]
        curRows = img.shape[1]

        IMG[0:curRows, iCol:(iCol + curCols),:] = img
        iCol += curCols


    # for


    pass

def colorImg(img, clr):

    if clr == 'r':
        axis = 0
    elif clr == 'g':
        axis = 1
    elif clr == 'b':
        axis = 2
    else:
        print "wtf?"
        raise

    notAxis = [0,1,2].remove(axis)

    img[0,:,axis]=255
    img[-1,:,axis]=255
    img[:,0,axis]=255
    img[:,-1,axis]=255

    for ax in notAxis:
        img[0,:,ax]=0
        img[-1,:,ax]=0
        img[:,0,ax]=0
        img[:,-1,ax]=0

    return img