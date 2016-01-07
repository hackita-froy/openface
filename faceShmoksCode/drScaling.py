import matplotlib.pyplot as plt
import skimage as ski
import numpy as np
from skimage import exposure
import loopTimer as lt
import os
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def walkAndRescale(root_dir):
    nFiles = 0
    for root, dirs, files in os.walk(root_dir, topdown=True):
        nFiles += len([f for f in files if f.lower().endswith(('.jpg', '.png', '.tif'))])


    ts = lt.resetTimer(nFiles,'Rescaling dr in images!', byIterOrTime='time', dt=5)

    iFile = 0

    for root, dirs, files in os.walk(root_dir, topdown=True):

        ent = root.split(root_dir)[-1]
        if ent.startswith(r'/'):
            ent = ent[1:]

        print "-----"  + ent + "-----" #+ str(ent.lower().startswith(r'ie'))

        for name in files:


            key = os.path.join(ent, name)

            fname = os.path.join(root, name)

            if not fname.endswith(('.jpg', '.png', '.tif')):

                continue

            try:
                img = plt.imread(fname)
            except:
                print "error reading: ", fname, ", continuing"
                continue

            if len(img.shape)<=2:
                continue

            p2, p98 = np.percentile(img, (2, 98))
            img_rescale = exposure.rescale_intensity(rgb2gray(img), in_range=(p2, p98))

            imgObj = Image.fromarray((img_rescale * 255).astype(np.uint8))

            imgObj.save(fname)

            iFile += 1

            lt.sampleTimer(iFile ,ts)



def main():
    img = plt.imread('/home/michael/data/nli_faces_drScaling/IE12094281/FL12095063.jpg')
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(rgb2gray(img), in_range=(p2, p98))

    plt.ion()
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(img_rescale, cmap='gray')


if __name__ == '__main__':
    # main()
    walkAndRescale('/home/michael/data/nli_faces_drScaling')


