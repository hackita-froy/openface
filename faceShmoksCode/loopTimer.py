import time

def resetTimer(nIter, tag, percentile=10.0):

    startTime = time.time()

    assert type(nIter) == type(1)
    assert type(tag) == type('str')

    show_freq = max(round(percentile/100.0*nIter),1)

    return {'startTime':startTime, 'nIter':float(nIter), 'tag':tag, 'show_freq':show_freq}

def sampleTimer(iIter, timerDict):

    assert type(iIter) == type(1)

    iIter = iIter + 1

    curTime = time.time()
    startTime = timerDict['startTime']
    nIter = timerDict['nIter']
    show_freq = timerDict['show_freq']

    elpsTime = curTime - startTime

    totTime = elpsTime/iIter*nIter

    togoTime = totTime - elpsTime

    if iIter % show_freq == 0:
        print timerDict['tag'] + \
              " done: {0} out of {1} ({2:3.2%})".format(iIter, nIter, float(iIter)/nIter) + \
              ", elapsed:{0:.0f}s, to go:{1:.0f}s, tot projected:{2:.0f}s".format(round(elpsTime), round(togoTime), round(totTime))