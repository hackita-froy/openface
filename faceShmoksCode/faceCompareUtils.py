import numpy as np
import pickle as pkl
import os
import loopTimer
import matplotlib.pyplot as plt
from collections import Counter


def isVector(x):

    return (len(x.shape)==1 or (len(x.shape)==2 and min(x.shape)==1))


def l2Dist(x,y=None):

    hasY = True

    if type(y)==type(None):

        y=x
        hasY = False

    assert type(x) == np.ndarray and type(y)==np.ndarray, 'x and y must be of type numpy.ndarray'
    assert len(x.shape)==2 and len(y.shape)==2, "x and y must be matrices, i.e.: x.shape and y.shape must be of length 2"
    assert x.shape[1]==y.shape[1], "x and y must have same number of columns"

    d = x.shape[1]
    n = x.shape[0]
    m = y.shape[0]

    normSqX = (x*x).sum(1)
    if hasY:
        normSqY = (y*y).sum(1)
    else:
        normSqY = normSqX.copy()

    normSqX.resize((n,1))
    normSqY.resize((1,m))

    return normSqX + normSqY - 2*np.dot(x, y.T)




def getMatches(repsArr, queryRep, cmpFun = l2Dist, k=5):

    assert type(repsArr) == np.ndarray, 'repsArr must be of type numpy.ndarray'
    assert type(queryRep) == np.ndarray, 'queryRep must be of type numpy.ndarray'
    assert len(repsArr.shape)==2, "repsArr must be a matrix, i.e.: repsArr.shape must be of length 2"
    assert isVector(queryRep), "queryRep must be a vector, i.e.: queryRep.shape must be of length 1 or of length 2 with minimal dim 1"
    assert repsArr.shape[1]==min(queryRep.shape), "length of queryRep and # of columns in repsArr must be the same"

    d = repsArr.shape[1]
    n = repsArr.shape[0]

    queryRep = queryRep.reshape((1,d))

    # diff = repsArr - queryRep
    #
    # scores = (diff*diff).sum(1)

    scores = cmpFun(queryRep, repsArr).reshape((repsArr.shape[0]))

    nbrIdx = np.argsort(scores)[:k]

    return nbrIdx, scores[nbrIdx[:k]]


def dictToArr(D):

    dKeys = D.keys()

    arr = np.ndarray((len(dKeys),len(D[dKeys[0]])))

    for ii in xrange(len(dKeys)):
        arr[ii,:] = D[dKeys[ii]]

    return arr, dKeys

def produceReport(dbFname, qFname, cmpFun, savePath):

    print "aint't finished"
    return


    if not os.path.isdir(savePath):
        os.makedirs(savePath)

    isEmpty = len(os.listdir(savePath))==0

    if not isEmpty:
        yn=''
        while not (yn.lower().startswith('y') or yn.lower().startswith('n')):
            yn = raw_input('dir ' + savePath + ' is not empty!!! continue? [y/n] ')
        if yn.lower().startswith('n'):
            return

    D = pkl.load(open(dbFname,'r'))
    Q = pkl.load(open(qFname,'r'))

    DArr, DKeys = dictToArr(D)
    qArr, qKeys= dictToArr(Q)

    lt = loopTimer.resetTimer(len(Q),'comparing faces!',percentile=1.0,byIterOrTime='time', dt=1.)

    for ii in xrange(len(qKeys)):

        k, scores = cmpFun(DArr, qArr[ii,:])



        loopTimer.sampleTimer(ii, lt)


def testPerformance(DB, savePath, cmpFun = l2Dist, k=3000, numTh = 100, removeSingletons = True):

    if type(DB)==str:

        D = pkl.load(open(DB,'r'))

    elif type(DB)==dict:

        D = DB

    else:

        print "DB must be either a string or a dict"

        raise

    DArr, DKeys = dictToArr(D)

    ents = [key.split('/')[0] for key in DKeys]


    counts = Counter()

    for ent in ents:

        counts[ent]+=1


    if removeSingletons:

        # print counts

        NonSingletons = np.array([counts[ent] > 1 for ent in ents])

        DArr = DArr[NonSingletons,:]
        # DKeys = DKeys[NonSingletons]
        DKeys = [DKeys[ii] for ii in range(len(D)) if NonSingletons[ii]]
        ents = [ents[ii] for ii in range(len(D)) if NonSingletons[ii]]

        D = {key:D[key] for key in DKeys}

        # for k in DKeys:
        #     if counts[k]<=1:
        #         D.pop(k,None)

        # ynt len(D), len(DKeys), DArr.shape, len(ents)

    N = len(D)

    entMat = np.ndarray((N,N),dtype=type(True))


    for ii in xrange(N):
        for jj in xrange(N):
            entMat[ii,jj] = ents[ii]==ents[jj]


    scoreMat = cmpFun(DArr)

    SMat = entMat
    DMat = np.logical_not(entMat)

    S = float(SMat.sum())
    D = float(DMat.sum())

    thresh = np.linspace(0,scoreMat.max(),num=numTh)

    TP = np.ndarray((numTh))
    FP = np.ndarray((numTh))

    lt = loopTimer.resetTimer(numTh,'computing ROC',dt=1, byIterOrTime='time')
    for ii, th in enumerate(thresh):
        TP[ii] = np.logical_and(SMat,scoreMat <= th).sum()/S
        FP[ii] = np.logical_and(DMat,scoreMat <= th).sum()/D
        loopTimer.sampleTimer(ii,lt)

    # idx = np.argsort(scoreMat.ravel())
    #
    # TP = np.cumsum(SMat.ravel()[idx])/S
    # FP = np.cumsum(DMat.ravel()[idx])/D




    plt.ion()
    plt.figure()
    plt.plot(FP,TP)
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid()

    if not removeSingletons:
        return

    print "Computing match stats"

    matchRateMat = np.ndarray((N,k-1))


    lt=loopTimer.resetTimer(N,'computing retrieval stats', dt=1, byIterOrTime='time')
    for ii in xrange(N):
        curEnt = ents[ii]

        nbrIdx, scores = getMatches(DArr,DArr[ii,:],cmpFun=cmpFun,k=k)
        nbrIdx = nbrIdx[1:]
        scores = scores[1:]

        matchTF = np.array([curEnt==ents[idx] for idx in nbrIdx])
        # print matchTF

        expCounts = np.minimum(np.array(range(k-1))+1, float(counts[curEnt]))

        matchRate = np.cumsum(matchTF)/expCounts

        matchRateMat[ii,:] = matchRate

        loopTimer.sampleTimer(ii,lt)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.average(matchRateMat,axis=0))
    plt.ylabel('average retrieval performance')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(np.std(matchRateMat,axis=0))
    plt.ylabel('std of retrieval performance')
    plt.xlabel('k')
    plt.grid()

    plt.figure()
    plt.plot(matchRateMat.T)
    plt.grid()



def main1():


    dbFname = '/home/michael/data/nli_faces/repsDBase.pkl'

    qFname = '/home/michael/data/nli_faces_part/repsDBase.pkl'
    qFname = '/home/michael/data/zalmania_images/repsDBase.pkl'

    dbFname =   '/home/michael/data/nli_faces_part/repsDBase.pkl'
    qFname = '/home/michael/data/nli_faces_part/repsDBase.pkl'

    dbFname = '/home/michael/data/nli_faces/repsDBase.pkl'
    qFname = '/home/michael/data/nli_faces/repsDBase.pkl'

    D = pkl.load(open(dbFname,'r'))
    Q = pkl.load(open(qFname,'r'))

    schwArr, schwKeys = dictToArr(D)
    qArr, qKeys= dictToArr(Q)

    RES = {}

    RES['source'] = dbFname
    RES['queries'] = qFname


    lt = loopTimer.resetTimer(len(Q),'comparing faces!',percentile=1.0)

    for ii in xrange(len(qKeys)):

        k, scores = getMatches(schwArr, qArr[ii, :])

        nbr = 1

        RES[qKeys[ii]] = (schwKeys[k[nbr]], scores[nbr])

        # print "query: {0}, closest match: {1}, score: {2}".format(qKeys[ii],schwKeys[k[nbr]], scores[nbr])

        loopTimer.sampleTimer(ii, lt)

    pkl.dump(RES, open('/home/michael/data/cmpRes2.pkl','w'))

def main2():

    dbFname = '/home/michael/data/nli_faces/repsDBase.pkl'
    # dbFname = '/home/michael/data/lfw/repsDBase.pkl'



    testPerformance(dbFname, '')

if __name__ == "__main__":

    main2()

