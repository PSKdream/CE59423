import numpy as np
import cv2
import matplotlib.pyplot as plt
# import skimage
from skimage.measure import label,regionprops

A = cv2.imread('BB.jpg')
img = A.copy()
A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

# A = cv2.threshold(A,155,255,cv2.THRESH_BINARY)[1]
# print(A.shape)
A[:200] = A[:200] > 135
A[200:] = A[200:] > 110

B = np.ones((0,0))
C = cv2.erode(A,B)
# C = cv2.dilate(C,B)
D = cv2.dilate(C,B)
border = D - C


plt.subplot(1,3,1)
plt.imshow(A,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(C,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(border,cmap='gray')
plt.show()
L = label(C)
# plt.imshow(L)
# plt.show()
props = regionprops(L)
print('props size : ', len(props))
sumN = 0
for i in props:
    sumN += i['axis_major_length']
avg = sumN/len(props)
# print(avg)

broken = 0
for i in props:
    if i['axis_major_length'] < avg:
        broken += 1

    # Radius of circle
    radius = 2
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of -1 px
    thickness = -1
    center_coordinates = (int(i['centroid'][0]), int(i['centroid'][1]))
    img = cv2.circle(img, center_coordinates, radius, color, thickness)
    # img = cv2.circle(img, (int(i['centroid'][0]), int(i['centroid'][1])), radius, color, thickness)
    # start_point = i['bbox'][:2]
    # end_point = i['bbox'][2:]
    # img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1)

print(broken)



# image = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
cv2.imshow("window_name", img)
cv2.waitKey()
data = {}
for i in props:
    if i['area'] in data:
        data[i['area']] += 1
    else:
        data[i['area']] = 1

print(sum(data.values()))
print(data)

plt.imshow(L)
plt.show()