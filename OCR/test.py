import cv2
import os
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("f1", frame)
    key = cv2.waitKey(50)
    if key == ord(' '):  # 空格跳出循环，结束程序
        break
    elif key == ord('x'):
        gay_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d_img = cv2.dilate(gay_img, None, iterations=4)
        e_img = cv2.erode(d_img, None, iterations=4)
        edges = cv2.absdiff(gay_img, e_img)
        cv2.imshow("f5", edges)
        x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)  # Sobel算子，X轴一阶导，Y轴不求导
        y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)  # Sobel算子，Y轴一阶导，X轴不求导
        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 将转换后的absX和absY组合起来，得到新的图像
        cv2.imshow("f6", dst)
        ret,th_img1 = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
        ret,th_img2 = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)
        cv2.imshow("f55", th_img1)
        cv2.imshow("f66", th_img2)
        cv2.imshow("f7", absX)
        cv2.imshow("f8", absY)

        new_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        print(frame.shape)
        print(new_edges.shape)

        cv2.waitKey(0)
#
# # a=np.array([1,3,2,1,3,4,1,6,5,4,5,6])
# # if a == None:
# #     pass

# knn = cv2.ml_KNearest.create()
# print(dir(knn))
