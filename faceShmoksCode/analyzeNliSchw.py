import getFaceRep
import logging
import os
import random
import sys
import pickle
import loopTimer as lt
# from pathlib import Path

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

dataPath = '/openface/data/nli_faces/'
processedFname = 'processed.pkl'
repFname = 'repsDBase.pkl'
failFname = 'failDBase.pkl'
startTime = None


def walkAndCalcReps(root_dir):

    """Clasify files in a given directory to prortrait or not.
    Traverses all sub directories of the given root
    Saves to db if portrait
    """
    global processedFname
    global repFname, failFname

    processedFname = os.path.join(dataPath,processedFname)
    repFname = os.path.join(dataPath,repFname)
    failFname = os.path.join(dataPath,failFname)

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

    dirs = [s for s in os.listdir(root_dir) if s.lower().startswith('ie')]

    # print '********'
    # print dirs
    # print '********'

    nDirs = len(dirs)
    iDir = 0

    ts = lt.resetTimer(nDirs,'Going over NLI dirs!')

    for root, dirs, files in os.walk(root_dir, topdown=True):

        # print "***" + root + "***"
        # print files

        ent = root.split('/')[-1]

        if not ent.lower().startswith(r'ie'):

            continue

        print "-----"  + ent + "-----" #+ str(ent.lower().startswith(r'ie'))

        for name in files:


            key = os.path.join(ent, name)
            # print key

            if key in processedList:

                continue

            # print "=====calculating rep====="

            fname = os.path.join(root, name)

            if not fname.endswith(('.jpg', '.png', '.tif')):

                continue

            rep, err = getImgRep(fname)

            if rep is not None:

                repDict[key]=rep

            else:

                failDict[key]=err

            processedList.append(key)

        pickle.dump(repDict,open(repFname,'w'),pickle.HIGHEST_PROTOCOL)

        pickle.dump(failDict,open(failFname,'w'))

        pickle.dump(processedList,open(processedFname,'w'))

        lt.sampleTimer(iDir,ts)

        iDir = iDir + 1


            #if is_port[0]:
                # repo.save_portrait_to_db(os.path.join(root, name), is_port[1])
                # logger.info("Portrait! %s" % os.path.join(root, name))
            # logger.error("NOT Portrait! %s" % os.path.join(root, name))


        # for name in dirs:
        #     print(os.path.join(root, name))

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

    walkAndCalcReps(dataPath)


    # for fname in fileList:
    #     print fname
    #     try:
    #         rep = getFaceRep.getRep(fname)
    #     except Exception as e:
    #         print "got this error:", str(e) #sys.exc_info()

    # firstShot('/openface/code/openface/')
