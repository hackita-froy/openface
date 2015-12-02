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
    print getFaceRep.getRep('//openface//code//openface//adams.jpg')
    # firstShot('/openface/code/openface/')
