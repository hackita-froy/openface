import numpy as np
import pickle as pkl
import os
import loopTimer
import matplotlib.pyplot as plt
from collections import Counter
import pdb
import metric_learn
from scipy.stats import hypergeom
import faceCompareUtils as fc
from metric_learn import ITML


def getConstraints(TD, k, thresh = 0.001):

    DArr, DKeys, ents, counts = fc.processDict(TD)

    nItems = len(TD)
    # k = len(k)
    # k = max(k)

    probScoresMat = np.ndarray((nItems, k))
    matchTFMat = np.ndarray((nItems, k), dtype=bool)
    nbrIdxMat = np.ndarray((nItems, k), dtype=int)

    lt = loopTimer.resetTimer(nItems, 'quering...', dt=1., byIterOrTime='time')
    for ii in xrange(nItems):

        curEnt = ents[ii]


        nbrIdx, scores = fc.getMatches(DArr,DArr[ii,:],k=k+1)
        nbrIdx = nbrIdx[1:]
        scores = scores[1:]
        nbrIdxMat[ii,:] = nbrIdx

        matchTFMat[ii,:] = np.array([curEnt==ents[idx] for idx in nbrIdx])

        # pdb.set_trace()

        probScoresMat[ii,:] = fc.calcProbScore(matchTFMat[ii,:], np.arange(1,k+1))

        loopTimer.sampleTimer(ii, lt)

    M = probScoresMat <= thresh

    termIdx = np.array([np.max(np.arange(k)[M[ii,:]]) if np.any(M[ii,:]) else np.nan for ii in xrange(nItems)])

    # pdb.set_trace()

    A, B, C, D = buildITMLContstr(matchTFMat, nbrIdxMat, termIdx)

    validateConstraints(A,B,C,D, ents)

    return A, B, C, D, DArr, DKeys, ents



def validateConstraints(A,B,C,D, ents):

    print "Validating constraints"

    L = [len(l) for l in [A, B, C, D]]

    assert max(L)==min(L)
    N = len(A)

    for ii in range(N):

        assert A[ii]==C[ii]

        assert ents[A[ii]] == ents[B[ii]]

        assert ents[C[ii]] != ents[D[ii]]


def buildITMLContstr(matchTFMat, nbrIdxMat, termIdx):

    print "Building constraints"

    # pdb.set_trace()

    A=[]
    B=[]
    C=[]
    D=[]

    nItems = len(termIdx)
    # nQueries = len(K)
    # maxK = max(K)
    k = matchTFMat.shape[1]

    nAnswers = sum(np.logical_not(np.isnan(termIdx)))
    iAnswer = 0

    gotViolations = np.ndarray(nAnswers, dtype=bool)

    assert nItems == len(termIdx) and nItems == matchTFMat.shape[0]

    for ii in range(nItems):

        if not np.isnan(termIdx[ii]):

            idx = int(termIdx[ii])

            answerLine = matchTFMat[ii, :]

            gotViolations[iAnswer] = any([not answerLine[jj] and answerLine[jj+1] for jj in range(min(idx,k-1))])

            posIdx = nbrIdxMat[ii, answerLine[:idx+1]]
            negIdx = nbrIdxMat[ii, np.logical_not(answerLine[:idx+1])]

            for P in posIdx:
                for N in negIdx:
                    A.append(ii)
                    B.append(P)
                    C.append(ii)
                    D.append(N)

            # pdb.set_trace()

            iAnswer += 1

    # pdb.set_trace()

    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)


    print "number of ents with violations: ", sum(gotViolations)

    print "number of answers for constraints: ", nAnswers

    print "total number of ents in constraints: ", len(set(A) | set(B) | set(C) | set(D))

    print "total number of constrints: ", len(A)

    return A, B, C, D


def getMtxs(TD, k, probTH,
            max_iters=1000,
            convergence_threshold=0.001,
            gamma_v = np.logspace(-2,2,10)):

    A, B, C, D, DArr, DKeys, ents = getConstraints(TD, k, probTH)

    # pdb.set_trace()

    M = {}

    for ii, gamma in enumerate(gamma_v):

        print "beginning reg #", ii, " out of ", len(gamma_v)
        print "gamma: ", gamma

        # pdb.set_trace()

        itml = ITML(gamma=gamma,
                    max_iters=max_iters,
                    convergence_threshold=convergence_threshold)

        constr = (A,B,C,D)

        itml.fit(DArr, constr, verbose=True)

        M[gamma]=(itml.metric())

        fname = os.path.join(r'/home/michael/results', 'mtx_gamma='+"{:.5g}".format(gamma)+'.npy')

        np.save(fname, itml.metric())

        pkl.dump(M, open('/home/michael/results/matrices.pkl','w'))

    return M




def main():

    dbFname = '/home/michael/data/nli_faces/repsDBase.pkl'
    # dbFname = '/home/michael/data/nli_faces_drScaling/repsDBase.pkl'
    # dbFname = '/home/michael/data/lfw/repsDBase.pkl'

    D = pkl.load(open(dbFname,'r'))

    trainDict, testDict = fc.splitTrainTest(D, .30, 14081979)

    # A, B, C, D = getConstraints(trainDict,40, 0.05)

    getMtxs(trainDict, 40, 0.05,
            max_iters=100,
            convergence_threshold=0.01,
            gamma_v = np.logspace(2,-2,10))



if __name__ == "__main__":

    main()
