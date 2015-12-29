import getFaceRep
import logging
import os
import random
import sys
import pickle
import loopTimer as lt
import json
import argparse

# from pathlib import Path

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# dataPath = '/openface/data/nli_faces/'


def walkAndCalcReps(root_dir):

    """Clasify files in a given directory to prortrait or not.
    Traverses all sub directories of the given root
    Saves to db if portrait
    """
    processedFname = os.path.join(root_dir,'processed.pkl')
    repFname = os.path.join(root_dir,'repsDBase.pkl')
    repFnameOldFmt = os.path.join(root_dir,'repsDBaseOldFmt.pkl')
    failFname = os.path.join(root_dir,'failDBase.pkl')
    repFnameJson = os.path.join(root_dir,'repsDBase.json')

    startTime = None

    # processedFname = os.path.join(root_dir,processedFname)
    # repFname = os.path.join(root_dir,repFname)
    # failFname = os.path.join(root_dir,failFname)

    processedList = []
    if os.path.isfile(processedFname):
        processedList = pickle.load(open(processedFname,'r'))

    repDict = {}
    if os.path.isfile(repFname):
        repDict = pickle.load(open(repFname,'r'))

    failDict = {}
    if os.path.isfile(failFname):
        failDict = pickle.load(open(failFname,'r'))



    # print repFname
    # print

    # dirs = [s for s in os.listdir(root_dir) if s.lower().startswith('ie')]

    # print '********'
    # print dirs
    # print '********'

    # nDirs = len(dirs)
    # iDir = 0

    # first walk - to get the number of files
    nFiles = 0
    for root, dirs, files in os.walk(root_dir, topdown=True):
        nFiles += len([f for f in files if f.lower().endswith(('.jpg', '.png', '.tif'))])


    ts = lt.resetTimer(nFiles,'Analyzing images!', byIterOrTime='time', dt=5)

    iFile = 0

    for root, dirs, files in os.walk(root_dir, topdown=True):

        # print "***" + root + "***"
        # print files

        ent = root.split(root_dir)[-1]
        if ent.startswith(r'/'):
            ent = ent[1:]

        print "-----"  + ent + "-----" #+ str(ent.lower().startswith(r'ie'))

        for name in files:


            key = os.path.join(ent, name)
            # print key

            # if key in processedList:
            #
            #     continue

            # print "=====calculating rep====="

            fname = os.path.join(root, name)

            if not fname.endswith(('.jpg', '.png', '.tif')):

                continue

            rep, err = getImgRep(fname)

            iFile += 1

            lt.sampleTimer(iFile ,ts)


            if rep is not None:

                repDict[key]=rep

            else:

                failDict[key]=err

            processedList.append(key)

        pickle.dump(repDict,open(repFname,'w'),pickle.HIGHEST_PROTOCOL)
        pickle.dump(repDict,open(repFnameOldFmt, 'w'))

        pickle.dump(failDict,open(failFname,'w'))

        pickle.dump(processedList,open(processedFname,'w'))

    repDictJson = {k:list(v) for k,v in repDict.iteritems()}

    f = open(repFnameJson,'w')
    f.write(json.dumps(repDictJson))
    f.close()



def getImgRep(fname):
    """Return a tuple (is_portrait, portrait_bounding_box)"""

    # add image formats as needed

    rep = None
    err = ''

    # if fname.endswith(('.jpg', '.png', '.tif')):
    try:
        rep = getFaceRep.getRep(fname)
    except Exception as e:
        pass
        err = 'error: ' + str(e) #sys.exc_info()

    return rep, err



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('root_dir', type=str, nargs=1, help="Absolute path to dir with images.")

    args = parser.parse_args()

    walkAndCalcReps(args.root_dir[0])

