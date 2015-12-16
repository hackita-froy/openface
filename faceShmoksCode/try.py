import getFaceRep
import sys

# def firstShot(imgPath):
#     print ' qqq!'
#     try:
#         rep = getFaceRep.getRep(imgPath)
#     except:
#         print "Unexpected error:", sys.exc_info()[0]
#         raise


if __name__ == "__main__":

    fileList = ['/openface/code/openface/images/examples/adams.jpg',
                '/openface/code/openface/images/dlib-landmark-mean.png',
                'openface/code/openface/images/nn4.v1.conv1.lennon-1.png',
                'openface/code/openface/images/nn4.v1.lfw.roc.png',]

    for fname in fileList:
        print fname
        try:
            print getFaceRep.getRep(fname)
        except Exception as e:
            print "got this error:", str(e) #sys.exc_info()

    # firstShot('/openface/code/openface/')
