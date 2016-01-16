import os
import numpy as np
import matplotlib.pyplot as plt
from makeReport import makeReportImg, makeReportText
import faceCompareUtils as fc
import pickle as pkl
import pdb
import  loopTimer as lt
import faceCompareUtils as fc


def getResponsesToSelf(dbFname, outputPath, K):

    max_k = max(K)

    D = pkl.load(open(dbFname,'r'))
    title = dbFname.split('/')[-2]
    outputPath = os.path.join(outputPath, title)
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    queryPath = os.path.join('/', *tuple(dbFname.split('/')[:-1]))

    fc.removeSingletons(D)
    repsArr, DKeys, ents, counts = fc.processDict(D)

    rawFnames = genFnames(DKeys)

    # print queryFnames
    # print rawFnames

    N = len(D)



    timer = lt.resetTimer(N, 'computing resposnses',dt=3,byIterOrTime='time')

    for ii in xrange(N):

        curEnt = ents[ii]

        nbrIdx, scores = fc.getMatches(repsArr, repsArr[ii,:], k=10000)

        assert scores[0] < 1e-6, "score too large for self-match!!!"

        nbrIdx = nbrIdx[1:]

        scores = scores[1:]

        matchTF = np.array([curEnt==ents[idx] for idx in nbrIdx])

        matchTF4RprtImg, matchesFnames = getMatchesFnamesAndLabels(DKeys, matchTF, max_k, nbrIdx)

        expMathces = np.array([min(k, float(counts[curEnt]-1)) for k in K], dtype=float)
        actMatches = np.array([sum(matchTF[:k]) for k in K], dtype=float)

        corrMatchPrcnt = actMatches / expMathces*100.
        # pdb.set_trace()
        corrMatchPrcntStr = reduce(lambda a,b:a+', '+b,map(lambda s: "{0:.6g}".format(s),corrMatchPrcnt))


        imgOutputFname = os.path.join(outputPath, curEnt + "_" + rawFnames[ii])

        print '-----' + os.path.join(queryPath, DKeys[ii] ) +'-----'

        IMG = makeReportImg(queryPath, DKeys[ii],
                          queryPath, matchesFnames,
                          labels=matchTF4RprtImg)
        plt.imsave(imgOutputFname, IMG)


        titlesLine, keysLine, scoresLine = makeReportText(queryPath, matchesFnames, scores[:max_k+1])

        probScores = fc.calcProbScore(matchTF,K)
        probScoresStr = reduce(lambda a,b:a+', '+b,map(lambda s: "{0:.6g}".format(s),probScores))


        a,b = os.path.splitext(imgOutputFname)
        txtOutputFname = a + '.txt'
        with open(txtOutputFname,'w') as f:
            f.write(titlesLine + '\n')
            f.write(keysLine + '\n')
            f.write(scoresLine + '\n')
            f.write(probScoresStr + '\n')
            f.write(corrMatchPrcntStr + '\n')


        # pdb.set_trace()


        lt.sampleTimer(ii, timer)


def getMatchesFnamesAndLabels(DKeys, matchTF, max_k, nbrIdx):
    noCloseMatch = not any(matchTF[:max_k + 1])
    if noCloseMatch:
        # pdb.set_trace()

        idx = (np.arange(len(nbrIdx))[matchTF])[0]

        firstMatchIdx = nbrIdx[idx]

        assert matchTF[idx]
    nbrIdx4RprtImg = nbrIdx[:max_k + 2]
    matchTF4RprtImg = matchTF[:max_k + 2]
    if noCloseMatch:
        nbrIdx4RprtImg[-1] = firstMatchIdx
        matchTF4RprtImg[-1] = True

    else:

        nbrIdx4RprtImg = nbrIdx[:max_k + 1]
        matchTF4RprtImg = matchTF[:max_k + 1]
    matchesFnames = [DKeys[nbr] for nbr in nbrIdx4RprtImg]
    return matchTF4RprtImg, matchesFnames


def genFnames(DKeys):
    rawFnames = []
    for key in DKeys:

        curImgName = key.split('/')[1]

        rawFnames.append(curImgName)


    return rawFnames



def genQFnames(DKeys):


    rawFnames = []

    for key in DKeys:

        if key.find('/')>0:

            curImgName = key.split('/')[-1]

        else:

            curImgName = key

        rawFnames.append(curImgName)

        # queryFnames.append(os.path.join(ent, curImgName))

    return  rawFnames


def getResponsesToQuery(dbFname, qFname, outputPath, k=20):

    title = dbFname.split('/')[-2] + "_" + qFname.split('/')[-2]

    outputPath = os.path.join(outputPath, title)

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    dbPath = os.path.join('/', *tuple(dbFname.split('/')[:-1]))
    queryPath = os.path.join('/', *tuple(qFname.split('/')[:-1]))

    D = pkl.load(open(dbFname,'r'))
    repsArrD, DKeysD, entsD, countsD = fc.processDict(D)

    Q = pkl.load(open(qFname,'r'))
    repsArrQ, DKeysQ, junk, junk = fc.processDict(Q)


    # pdb.set_trace()



    N = len(Q)

    timer = lt.resetTimer(N, 'computing resposnses',dt=3,byIterOrTime='time')


    for ii in xrange(N):

        # curEnt = entsQ[ii]

        nbrIdx, scores = fc.getMatches(repsArrD, repsArrQ[ii,:], k=20)
        matchesFnames = [DKeysD[nbr] for nbr in nbrIdx]

        IMG = makeReportImg(queryPath, DKeysQ[ii],
                          dbPath, matchesFnames,
                          labels=None)

        titlesLine, keysLine, scoresLine = makeReportText(dbPath, matchesFnames, scores)

        imgOutputFname = os.path.join(outputPath,  DKeysQ[ii])

        dumpDir = os.path.join('/',*tuple(imgOutputFname.split('/')[:-1]))
        if not os.path.isdir(dumpDir):
            os.makedirs(dumpDir)

        assert not os.path.isfile(imgOutputFname), "output file " + imgOutputFname + " already exists!"

        plt.imsave(imgOutputFname, IMG)

        a,b = os.path.splitext(imgOutputFname)
        txtOutputFname = a + '.txt'
        with open(txtOutputFname,'w') as f:
            f.write(titlesLine + '\n')
            f.write(keysLine + '\n')
            f.write(scoresLine + '\n')


        lt.sampleTimer(ii, timer)

def main():

    dbasePath = '/home/michael/data/nli_faces_part/repsDBase.pkl'
    dbasePath = '/home/michael/data/nli_faces/repsDBase.pkl'
    dbasePath = '/home/michael/data/lfw/repsDBase.pkl'


    outputPath = '/home/michael/results'

    getResponsesToSelf(dbasePath, outputPath, K = np.array([5,10,20,40]))

def sampleQ():

    dbasePath = '/home/michael/data/nli_faces/repsDBase.pkl'
    qPath = '/home/michael/data/sample_query/repsDBase.pkl'

    outputPath = '/home/michael/results'

    getResponsesToQuery(dbasePath, qPath, outputPath, k=20)

if __name__ == "__main__":

    main()
