import cv2 
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_histogram(image, bins):
    
    histogram = np.zeros(bins)

    for pixel in image:
        histogram[pixel] += 1
    return histogram


def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

def Normalize(cs):
    
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    cs = nj / N
    return cs

def Histogram(image):

    # r = image[:,:,2]
    # g = image[:,:,1]
    # b = image[:,:,0]

    image = np.array(image)
    # imgr = np.asarray(r)
    # imgg = np.asarray(g)
    # imgb = np.asarray(b)

    flat = image.flatten()
    # flatr = imgr.flatten()
    # flatg = imgg.flatten()
    # flatb = imgb.flatten()

    hist = get_histogram(flat, 256)

    # histr = get_histogram(flatr, 256)
    # histg = get_histogram(flatg, 256)
    # histb = get_histogram(flatb, 256)

    cs = cumsum(hist)

    # csr = cumsum(histr)
    # csg = cumsum(histg)
    # csb = cumsum(histb)

    cs = Normalize(cs)

    # csr = Normalize(csr)
    # csg = Normalize(csg)
    # csb = Normalize(csb)

    cs = cs.astype('uint8')

    # csr = csr.astype('uint8')
    # csg = csg.astype('uint8')
    # csb = csb.astype('uint8')

    img_new = np.zeros((image.shape))

    # img_newr = np.zeros((image.shape[1],image.shape[0]))
    # img_newg = np.zeros((image.shape[1],image.shape[0]))
    # img_newb = np.zeros((image.shape[1],image.shape[0]))
    
    img_new = alpha * cs[flat] + ((1-alpha) * flat[flat])



    img_new = np.reshape(img_new, image.shape)

    img_new = img_new.astype('uint8')

    return img_new

def adaptive_equilize(img):
    img = img.copy()
    h, w = img.shape
    # h, w,_ = img.shape
    bh, bw = h//8, w//8

    for i in range(8):
        for j in range(8):
            img[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = Histogram(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw])
            img = np.array(img, dtype=np.uint8)
            # img = cv2.medianBlur(img, 3)
    return img


path = glob.glob(r"C:\Users\chand\Downloads\adaptive_hist_data-20220404T031542Z-001\adaptive_hist_data\*")
alpha = 1

for file in path:
    image = cv2.imread(file)
    frame_width = image.shape[1]
    frame_height = image.shape[0]

    imager = image[:,:,2]
    imageg = image[:,:,1]
    imageb = image[:,:,0]

    img_histr = Histogram(imager)
    img_histg = Histogram(imageg)
    img_histb = Histogram(imageb)

    img_histogram = np.zeros((img_histr.shape[0],img_histr.shape[1],3), np.uint8)

    img_histogram[:,:,2] = img_histr
    img_histogram[:,:,1] = img_histg
    img_histogram[:,:,0] = img_histb

    img_newr = adaptive_equilize(imager)
    img_newg = adaptive_equilize(imageg)
    img_newb = adaptive_equilize(imageb)

    img_adaptive = np.zeros((img_newr.shape[0],img_newr.shape[1],3), np.uint8)

    img_adaptive[:,:,2] = img_newr
    img_adaptive[:,:,1] = img_newg
    img_adaptive[:,:,0] = img_newb

    # Vertical = np.concatenate((img_histogram, img_adaptive), axis=0)
    # cv2.imshow('', Vertical )

    cv2.imshow('Histogram',img_histogram)
    cv2.imshow('Adaptive Histogram',img_adaptive)
    
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cv2.destroyAllWindows()
