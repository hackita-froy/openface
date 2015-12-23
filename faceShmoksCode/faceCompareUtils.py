import numpy as np

def isVector(x):

    return (len(x.shape)==1 or (len(x.shape)==2 and min(x.shape)==1))


def getClosestNbrs(repsArr, queryRep, k=5):

    assert type(repsArr) == np.ndarray, 'repsArr must be of type numpy.ndarray'
    assert type(queryRep) == np.ndarray, 'queryRep must be of type numpy.ndarray'
    assert len(repsArr.shape)==2, "repsArr must be a matrix, i.e.: repsArr.shape must be of length 2"
    assert isVector(queryRep), "queryRep must be a vector, i.e.: queryRep.shape must be of length 1 or of length 2 with minimal dim 1"
    assert repsArr.shape[1]==min(queryRep.shape), "length of queryRep and # of columns in repsArr must be the same"

    d = repsArr.shape[1]
    n = repsArr.shape[0]

    queryRep = queryRep.reshape((1,d))

    diff = repsArr - queryRep

    norms = np.sqrt((diff*diff).sum(1))

    nbrIdx = np.argsort(norms)[:k]

    return nbrIdx, norms[nbrIdx[:k]]


def dictToArr(D):

    dKeys = D.keys()

    arr = np.ndarray((len(dKeys),len(D[dKeys[0]])))

    for ii in xrange(len(dKeys)):
        arr[ii,:] = D[dKeys[ii]]

    return arr, dKeys


def main():

    import pickle as pkl

    dbFname = '/home/michael/data/nli_faces/repsDBase.pkl'

    qFname = '/home/michael/data/nli_faces_part/repsDBase.pkl'

    D = pkl.load(open(dbFname,'r'))
    Q = pkl.load(open(qFname,'r'))

    schwArr, schwKeys = dictToArr(D)
    qArr, qKeys= dictToArr(Q)


    for ii in xrange(len(qKeys)):

        k, scores = getClosestNbrs(schwArr, qArr[ii,:])

        print "query: {0}, closest match: {1}, score: {2}".format(qKeys[ii],schwKeys[k[0]], scores[0])



if __name__ == "__main__":

    main()

