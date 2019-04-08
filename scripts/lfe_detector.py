import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 20
template = cv.imread('../img/upside.jpeg', cv.IMREAD_COLOR)
test_img = cv.imread('../img/test3.jpg', cv.IMREAD_COLOR)


# def load_camera():
#     cap = cv.VideoCapture(0)
#     ret, frame = cap.read()
#     return canny_edge(frame)


def detect_circles(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=140, param2=20,
                              minRadius=30, maxRadius=300)
    if circles is not None:
        return np.uint16(np.around(circles))
    else:
        return None


def crop_circle_img(img, circle, margin):
    print(circle)
    x1 = circle[1] - circle[2] - margin
    x2 = circle[1] + circle[2] + margin
    y1 = circle[0] - circle[2] - margin
    y2 = circle[0] + circle[2] + margin
    cropped_img = img[x1:x2, y1:y2]
    return cropped_img


def sif_feature_detection(template, test_img):
    sift = cv.xfeatures2d.SURF_create()

    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(test_img, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)


    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return len(good)


def draw_circles(img, circles):
    for i in circles[0, :]:
        center = (i[0], i[1])

        circle_img = crop_circle_img(test_img, i, 30)

        cv.circle(img, center, 1, (0, 255, 0), 3)
        radius = i[2]

        print(sif_feature_detection(template, circle_img))

        if sif_feature_detection(template, circle_img) > MIN_MATCH_COUNT:
            cv.circle(img, center, radius, (255, 0, 0), 3)
        else:
            cv.circle(img, center, radius, (0, 255, 0), 3)

        # circle center
        # circle outline
    return img




circles = detect_circles(test_img)

plt.imshow(draw_circles(test_img, circles), 'gray'), plt.show()








# When everything done, release the capture
cv.destroyAllWindows()

# if len(good) > MIN_MATCH_COUNT:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#
#     h, w = template.shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#     dst = cv.perspectiveTransform(pts, M)
#
#     img2 = cv.polylines(test_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
#
# else:
#     print
#     "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
#     matchesMask = None
#
# draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                    singlePointColor=None,
#                    matchesMask=matchesMask,  # draw only inliers
#                    flags=2)
#
# img3 = cv.drawMatches(template, kp1, test_img, kp2, good, None, **draw_params)
#
# plt.imshow(img3, 'gray'), plt.show()