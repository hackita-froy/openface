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
    fnames.append(os.path.join(queryPath, queryFname))
    for m in matchesFnames:
        fnames.append(os.path.join(matchesPath, m))

    I = []
    for f in fnames:
        assert os.path.isfile(f), "cannot find file: " + f
        img = plt.imread(f)
        # print len(img.shape)
        if len(img.shape)==2:
            # print "bad sape"
            rgbImg = np.ndarray(img.shape + (3,), dtype=img.dtype)
            rgbImg[:,:,0]=img
            rgbImg[:,:,1]=img
            rgbImg[:,:,2]=img
            img = rgbImg

        I.append(img)

    shapes_ok = [len(img.shape) == 3 and img.shape[2] == 3 for img in I]
    types_ok = [img.dtype == np.dtype(np.uint8) for img in I]
    allRGB = all(shapes_ok) and \
             all(types_ok)

    if not allRGB:
        print  "eehhmmm... not images read are rgb... fix this plz..."
        print 'shapes: ', shapes_ok
        print 'types: ', types_ok
        print "imgs with bad shape:", [fnames[ii] for ii in xrange(len(fnames)) if not shapes_ok[ii]]
        print "imgs with bad type:", [fnames[ii] for ii in xrange(len(fnames)) if not types_ok[ii]]
        raise

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

        curRows = img.shape[0]
        curCols = img.shape[1]

        IMG[0:curRows, iCol:(iCol + curCols),:] = img
        iCol += curCols

    return IMG



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

    notAxis = [0,1,2]
    notAxis.remove(axis)

    bw_row = img.shape[0]/20
    bw_col = img.shape[1]/20

    img[0:bw_row,:,axis]=255
    img[-bw_row:,:,axis]=255
    img[:,0:bw_col,axis]=255
    img[:,-bw_col:,axis]=255

    for ax in notAxis:
        img[:bw_row,:,ax]=0
        img[-bw_row:,:,ax]=0
        img[:,:bw_col,ax]=0
        img[:,-bw_col:,ax]=0

    return img


def makeReportText(matchesPath, matchesFnames,
                   scores):

    # assert type(queryPath)==str and type(queryFname)==str and type(matchesPath)==str, \
    #     "queryPath, queryFname, matchesPath must be strings"

    assert type(matchesFnames)==list and all([type(q)==str for q in matchesFnames]), \
        "matchesFnames must be a list of strings"

    K = len(matchesFnames)

    fnames = []
    # fnames.append(os.path.join(queryPath, queryFname))
    for m in matchesFnames:
        fnames.append(os.path.join(matchesPath, m))

    titles = []
    for fname in fnames:
        assert os.path.isfile(fname), "cannot find file: " + fname
        p,f = os.path.split(fname)
        txtFname = os.path.join(p, 'title.text')
        assert os.path.isfile(txtFname), "cannot find file: " + txtFname

        f = open(txtFname,'r')
        line = f.read()
        f.close()
        line = line.replace(',','_')

        titles.append(line)

    titlesLine = reduce(lambda x,y:x+','+y, titles)
    keysLine = reduce(lambda x,y:x+','+y, matchesFnames)
    scoresLine = reduce(lambda x,y:x+','+y,['{:.6g}'.format(s) for s in scores])

    return titlesLine, keysLine, scoresLine



    # # I = []
    # for f in fnames:
    #     assert os.path.isfile(f), "cannot find file: " + f
    #     img = plt.imread(f)
    #     # print len(img.shape)
    #     if len(img.shape)==2:
    #         # print "bad sape"
    #         rgbImg = np.ndarray(img.shape + (3,), dtype=img.dtype)
    #         rgbImg[:,:,0]=img
    #         rgbImg[:,:,1]=img
    #         rgbImg[:,:,2]=img
    #         img = rgbImg
    #
    #     I.append(img)
    #
    # shapes_ok = [len(img.shape) == 3 and img.shape[2] == 3 for img in I]
    # types_ok = [img.dtype == np.dtype(np.uint8) for img in I]
    # allRGB = all(shapes_ok) and \
    #          all(types_ok)
    #
    # if not allRGB:
    #     print  "eehhmmm... not images read are rgb... fix this plz..."
    #     print 'shapes: ', shapes_ok
    #     print 'types: ', types_ok
    #     print "imgs with bad shape:", [fnames[ii] for ii in xrange(len(fnames)) if not shapes_ok[ii]]
    #     print "imgs with bad type:", [fnames[ii] for ii in xrange(len(fnames)) if not types_ok[ii]]
    #     raise
    #
    # nRows = max([img.shape[0] for img in I])
    # nCols = sum([img.shape[1] for img in I])
    #
    # IMG = np.ndarray((nRows,nCols,3),dtype = np.uint8)
    # IMG[...]=0
    #
    # iCol = 0
    #
    # for iImg, img in enumerate(I):
    #     if gotLabels:
    #         if iImg == 0:
    #             colorImg(img,'b')
    #         elif labels[iImg-1] == True:
    #             colorImg(img,'g')
    #         elif labels[iImg-1] == False:
    #             colorImg(img,'r')
    #
    #     curRows = img.shape[0]
    #     curCols = img.shape[1]
    #
    #     IMG[0:curRows, iCol:(iCol + curCols),:] = img
    #     iCol += curCols
    #
    # return IMG