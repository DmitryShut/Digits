# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2


def closiest_28(w, h):
    i = 28
    digit = 0
    if w > h:
        digit = w
    else:
        digit = h
    if digit < i:
        return i
    while i<digit:
        i+=i
    return i

def image_to_28_size(image):
    size = closiest_28(image.shape[0], image.shape[1])
    ar = np.full((size, size),255,image.dtype)
    highFrom = (int)((size - image.shape[0]) / 2)
    widthFrom = (int)((size - image.shape[1]) / 2)
    w = 0
    for i in range(len(ar)):
        h = 0
        for j in range(len(ar[i])):
            if (highFrom <= i) & (widthFrom <= j) & (h < image.shape[1]) & (w < image.shape[0]):
                ar[i][j] = image[w][h]
                h += 1
        if (highFrom <= i):
            w += 1
    return ar

# load and prepare the image
def load_image2(image, idx):
    grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image_to_28_size(grayimage), (28, 28), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("resized_{0}.png".format(idx), resized)
    resized = resized.reshape(1, 28, 28, 1)
    # prepare pixel data
    resized = resized.astype('float32')
    resized = resized / 255.0
    return resized

# load an image and predict the class
def run_example(image, idx):
    # load the image
    img = load_image2(image, idx)
    # load model
    model = load_model('final_model.h5')
    # predict the class
    digit = model.predict(img, 128, 0)
    digit = digit[0]
    strings = []
    for number in range(10):
        d = digit[number]
        if d > 0.1:
            strings.append(str(number) + "=" + str(d)[0:3])
    return strings