import time

def resetTimer(nIter, tag, percentile=10.0, dt=1.0, byIterOrTime='iter'):

    startTime = time.time()

    assert type(nIter) == type(1)
    assert type(tag) == type('str')

    assert type(byIterOrTime) == str
    byIterOrTime = byIterOrTime.lower()
    assert byIterOrTime=='iter' or byIterOrTime=='time'

    show_counts = max(round(percentile/100.0*nIter),1)

    return {'startTime':startTime,
            'nIter':float(nIter),
            'tag':tag,
            'show_counts':show_counts,
            'dt':float(dt),
            'byIterOrTime':byIterOrTime,
            'lastTime': startTime}


def sampleTimer(iIter, timerDict):

    # print iIter
    # print timerDict

    assert type(iIter) == type(1)

    iIter = iIter + 1

    curTime = time.time()
    startTime = timerDict['startTime']
    nIter = timerDict['nIter']
    show_counts = timerDict['show_counts']
    byIterOrTime = timerDict['byIterOrTime']
    lastTime = timerDict['lastTime']
    dt = timerDict['dt']

    elpsTime = curTime - startTime

    totTime = elpsTime/iIter*nIter

    togoTime = totTime - elpsTime

    msg = timerDict['tag'] + \
        " done: {0} out of {1} ({2:3.2%})".format(iIter, nIter, float(iIter)/nIter) + \
        ", elapsed:{0:.0f}s, to go:{1:.0f}s, tot projected:{2:.0f}s".format(round(elpsTime), round(togoTime), round(totTime))

    if byIterOrTime=='iter':

        if iIter % show_counts == 0:
            print msg

    elif byIterOrTime=='time':

        if curTime - lastTime >= dt:

            print msg
            timerDict['lastTime'] = curTime

        return  timerDict


