import cv2
import numpy as np
from skimage.measure import label, regionprops
import os

position = None
pts1 = []
pts2 = [[0, 0], [2480, 0], [2480, 1748], [0, 1748]]


def onClick(event, x, y, flags, param):
    global position
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts1) < 4:
            pts1.append([x, y])
            print(pts1)
    elif event == cv2.EVENT_MOUSEMOVE and len(pts1) >= 1:
        position = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(pts1) > 0:
            pts1.pop()
            # print(pts1)


cv2.namedWindow('resized')
cv2.setMouseCallback('resized', onClick)

for file in os.listdir("./img"):
    if file.endswith(".jpg"):
        print(os.path.join("./img", file))

        frame = cv2.imread(os.path.join("./img", file))
        scale_percent = 50  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image

        while True:
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            if len(pts1) == 4:
                T = cv2.getPerspectiveTransform(np.float32(pts1) * (scale_percent / 100)**-1, np.float32(pts2))
                img = cv2.warpPerspective(frame, T, (2480, 1748))
                cv2.imwrite(os.path.join("./out", file), img)
                pts1 = []
                position = None
                break
            elif position is not None and len(pts1) >= 1:
                for i in range(len(pts1) - 1):
                    cv2.line(resized, pts1[i], pts1[i + 1], (0, 255, 0), 1)

                cv2.line(resized, pts1[-1], position, (0, 255, 0), 1)

                if len(pts1) == 3:
                    cv2.line(resized, pts1[0], position, (0, 255, 0), 1)

            cv2.imshow('resized', resized)
            cv2.waitKey(1)
