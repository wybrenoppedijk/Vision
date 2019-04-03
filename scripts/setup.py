import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def canny_edge(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (500, 500))
    img_blur = cv.blur(img, (2, 2))
    return cv.Canny(img_blur, 50, 50)


def load_img(url):
    img = cv.imread(url)
    return canny_edge(img)


def load_camera():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    return canny_edge(frame)


# cap = cv.VideoCapture(0)
while True:
    upside_img = load_img('../img/upside.jpg')
    downside_img = load_img('../img/downside.jpg')
    test_img = load_img('../img/test_img2.jpg')

    sift = cv.xfeatures2d.SURF_create()

    frame = load_camera()

    kp1, des1 = sift.detectAndCompute(upside_img, None)
    kp2, des2 = sift.detectAndCompute(frame, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=90)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv.drawMatchesKnn(upside_img, kp1, frame, kp2, matches, None, **draw_params)

    cv.imshow('test', img3)
    # plt.imshow(img3, ), plt.show()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv.destroyAllWindows()
