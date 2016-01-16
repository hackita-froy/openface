import numpy as np
import pickle as pkl
import os
import loopTimer
import matplotlib.pyplot as plt
from collections import Counter
import pdb
import metric_learn
from scipy.stats import hypergeom


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


def processDict(D):

    DKeys = D.keys()

    arr = np.ndarray((len(DKeys), len(D[DKeys[0]])))

    for ii in xrange(len(DKeys)):
        arr[ii,:] = D[DKeys[ii]]

    ents = getEntsFromKeys(DKeys)

    counts = getCounts(ents)


    return arr, DKeys, ents, counts


def getCounts(ents):
    counts = Counter()
    for ent in ents:
        counts[ent] += 1
    return counts


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

    DArr, DKeys = processDict(D)
    qArr, qKeys= processDict(Q)

    lt = loopTimer.resetTimer(len(Q),'comparing faces!',percentile=1.0,byIterOrTime='time', dt=1.)

    for ii in xrange(len(qKeys)):

        k, scores = cmpFun(DArr, qArr[ii,:])



        loopTimer.sampleTimer(ii, lt)

def removeSingletons(D):

    DKeys = D.keys()

    ents = getEntsFromKeys(DKeys)

    counts = getCounts(ents)

    entAndKey = ((ents[ii], DKeys[ii]) for ii in range(len(DKeys)))


    for ent, key in entAndKey:
        if counts[ent] <= 1:
            # print ent, key, ent in counts, key in D
            del counts[ent]
            del D[key]

    return counts


def getEntsFromKeys(DKeys):

    ents = [key.split('/')[0] for key in DKeys]

    return ents


def testPerformance(DB, savePath,
                    cmpFun = l2Dist,
                    k=100,
                    numTh = 100,
                    mustRemoveSingletons = True,
                    title=''):

    if type(DB)==str:

        D = pkl.load(open(DB,'r'))
        if title=='':
            title = DB.split('/')[-2]


    elif type(DB)==dict:

        D = DB

    else:

        print "DB must be either a string or a dict"

        raise

    # DArr, DKeys = processDict(D)
    #
    # ents = [key.split('/')[0] for key in DKeys]


    if mustRemoveSingletons:

        removeSingletons(D)

    DArr, DKeys, ents, counts = processDict(D)


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



    plt.ioff()
    SMatNoSelf = SMat
    for ii in xrange(SMatNoSelf.shape[0]):
        SMatNoSelf[ii,ii]=False

    c2c = scoreMat.ravel()[SMat.ravel()]
    c2nc = scoreMat.ravel()[DMat.ravel()]

    plt.figure()
    plt.hist(c2c, bins=100, alpha = 0.4, normed=True)
    plt.hist(c2nc, bins=100, alpha = 0.4, normed=True)
    plt.legend(['c2c','c2nc'])
    plt.savefig(os.path.join(savePath, title + '_dist_hist.png'))

    plt.figure()
    plt.plot(FP,TP)
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid()
    plt.title(title + ": ROC")
    plt.savefig(os.path.join(savePath, title + '_ROC.png'))

    if not mustRemoveSingletons:
        return

    print "Computing match stats"

    k = k+1

    matchRateMat = np.ndarray((N,k-1))
    cumMatchesMat = np.ndarray((N,k-1))

    K=np.array([5,10,20,40])
    # K=[5]
    probScoresMat = np.ndarray((N, len(K)))

    FM = np.ndarray(N)

    CONSTR = ([],[],[],[])

    lt=loopTimer.resetTimer(N,'computing retrieval stats', dt=1, byIterOrTime='time')

    distsToMatch = []
    distToNonMatch = []

    for ii in xrange(N):
        # print ii
        curEnt = ents[ii]

        nbrIdx, scores = getMatches(DArr,DArr[ii,:],cmpFun=cmpFun,k=100000)
        nbrIdx = nbrIdx[1:]
        scores = scores[1:]

        matchTF = np.array([curEnt==ents[idx] for idx in nbrIdx])

        # pdb.set_trace()

        probScoresMat[ii,:] = calcProbScore(matchTF,K)
        # pdb.set_trace()
        # print matchTF

        expCounts = np.minimum(np.arange(k-1)+1, float(counts[curEnt]-1))

        cumMatches = np.cumsum(matchTF[:k-1])
        cumMatchesMat[ii,:] = cumMatches > 0

        matchRate = cumMatches/expCounts
        matchRateMat[ii,:] = matchRate[:k-1]

        haveMatch = np.any(matchTF)
        # ITML
        # [(Y[C[0][ii]],Y[C[1][ii]],Y[C[2][ii]],Y[C[3][ii]]) for ii in xrange(len(C[0]))]

        FM[ii] = np.nan
        if haveMatch:
            FM[ii] = (np.arange(matchTF.size)[matchTF])[0]


        loopTimer.sampleTimer(ii,lt)

    # pdb.set_trace()

    plt.figure()
    for i,kk in enumerate(K):
        plt.subplot(len(K), 1, i+1)
        plt.hist(probScoresMat[:,i].ravel(),100)
        plt.title('prob score for k=' + str(kk))
        plt.grid()
    plt.savefig(os.path.join(savePath, title + '_prob_scores.png'))


    x = np.arange(1,k)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x,np.average(matchRateMat,axis=0))
    plt.ylabel('average')
    plt.grid()
    plt.title(title + ": match rate")

    plt.subplot(3,1,2)
    plt.plot(x,np.std(matchRateMat,axis=0))
    plt.ylabel('std')
    plt.xlabel('k')
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(x,np.median(matchRateMat,axis=0))
    plt.ylabel('median')
    plt.xlabel('k')
    plt.grid()

    plt.savefig(os.path.join(savePath, title + '_AMR.png'))

    # pdb.set_trace()
    plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(x, cumMatchesMat.T)
    # plt.xlabel('k')
    # plt.ylabel('got 1 match?')
    #
    # plt.subplot(2,1,2)
    plt.plot(x, np.average(cumMatchesMat,axis=0))
    plt.xlabel('k')
    plt.ylabel('prob got 1 match?')
    plt.title(title + ": probability of having at least one match")
    plt.grid()
    plt.savefig(os.path.join(savePath, title + '_matchProb.png'))


    plt.figure()
    plt.plot(x,matchRateMat.T)
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('match rate, unaveraged')
    plt.title(title + ": raw retrieval performance")
    plt.savefig(os.path.join(savePath, title + '_MR.png'))


    FM = FM[np.logical_not( np.isnan(FM))]+1
    plt.figure()
    plt.hist(FM,bins=FM.max())
    plt.title(title + ": first match position distribution")
    plt.xlabel('First match position')
    plt.ylabel('counts')

    plt.savefig(os.path.join(savePath, title + '_FM.png'))

    plt.show()


def calcProbScore(matchTF, K):
    n = sum(matchTF)
    M = len(matchTF)

    p = np.ndarray(len(K))

    for ii,k in enumerate(K):

        d = sum(matchTF[:k])

        rv = hypergeom(M,n,k)

        p[ii] = rv.pmf(d)

    return p


def splitTrainTest(Dict, testPrcnt, seed):

    Dkeys = Dict.keys()

    ents = getEntsFromKeys(Dkeys)

    uniqueEnts = list(set(ents))
    uniqueEnts.sort()

    state = np.random.get_state()
    np.random.seed(seed)
    perm = np.random.permutation(len(uniqueEnts))
    np.random.set_state(state)

    divLoc = round(len(uniqueEnts)*testPrcnt)

    testIdx= perm[:divLoc]
    trainIdx = perm[divLoc:]

    testDict = extractDictByUniqueEntIdx(Dict, testIdx)
    trainDict = extractDictByUniqueEntIdx(Dict, trainIdx)

    return trainDict, testDict


def extractDictByUniqueEntIdx(Dict, subsIdx):

    Dkeys = Dict.keys()

    ents = getEntsFromKeys(Dkeys)

    uniqueEnts = list(set(ents))

    uniqueEnts.sort()

    subsEnts = [uniqueEnts[ii] for ii in subsIdx]

    subsMask = np.array([e in subsEnts for e in ents])

    subsKeys = [Dkeys[ii] for ii in range(len(Dkeys)) if subsMask[ii]]

    subsDict = {k: Dict[k] for k in subsKeys}

    return subsDict


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

    schwArr, schwKeys = processDict(D)
    qArr, qKeys= processDict(Q)

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
    # dbFname = '/home/michael/data/nli_faces_drScaling/repsDBase.pkl'
    dbFname = '/home/michael/data/lfw/repsDBase.pkl'



    testPerformance(dbFname, os.path.expanduser('~/results/'))

if __name__ == "__main__":

    main2()

