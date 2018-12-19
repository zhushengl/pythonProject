import cv2
import os
import numpy as np

THRESH = 20


def initKnn():
    knn = cv2.ml_KNearest.create()
    if os.path.exists("train.csv") and os.path.exists("train_label.csv"):
        train = np.loadtxt("train.csv", dtype=np.float32, delimiter=",")
        train_lable = np.loadtxt("train_label.csv", dtype=np.int32, delimiter=",")
    else:
        img = cv2.imread("digits.png")
        gay_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th_img = cv2.threshold(gay_img, THRESH, 255, cv2.THRESH_BINARY)
        cells = [np.hsplit(row, 100) for row in np.vsplit(th_img, 50)]
        train = np.array(cells).reshape(-1, 400).astype(np.float32)
        train_lable = np.repeat(np.arange(10), 500)
    return knn, train, train_lable


def updateKnn(knn, train, train_lable, newdata=None, newlable=None):
    if (newdata is not None) and (newlable is not None):
        # 将新数据维度变为(n,400),n行400列
        newdata = newdata.reshape(-1, 400).astype(np.float32)
        # 增加行的列数必须和原矩阵一致
        train = np.vstack((train, newdata))  # 垂直拼接(增加行)
        # train_lable（1，n）,newlable是列表，变换后也是(1,n)
        train_lable = np.hstack((train_lable, newlable))
    # 采用的ROW_SAMPLE行训练模式，即是说训练集中的每一行对应lable中的一列（1个数据）
    knn.train(train, cv2.ml.ROW_SAMPLE, train_lable)
    return knn, train, train_lable


def findRois(frame):
    gay_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 膨胀腐蚀，再相减（原灰度图和处理之后的图片）
    dilate = cv2.dilate(gay_img, None, iterations=4)
    erode = cv2.erode(dilate, None, iterations=4)
    edges = cv2.absdiff(gay_img, erode)
    # Sobel算子描边处理
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    # 将最后得到的图进行二值化处理
    ret, th_img = cv2.threshold(dst, THRESH, 255, cv2.THRESH_BINARY)
    th_img, contours, hierarchy = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 50 > w > 5 and 70 > h > 5:
            rois.append((x, y, w, h))
    rois.sort(key=lambda x: x[0])
    return rois, th_img


def findDigit(knn, img):
    new_img = cv2.resize(img, (20, 20))
    ret, s_img = cv2.threshold(new_img, THRESH, 255, cv2.THRESH_BINARY)
    samples = s_img.reshape(-1, 400).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(samples, k=5)
    return int(result[0][0]), s_img


def concatenate(img_list):
    size = len(img_list)
    result_img = np.zeros((20 * 20 * size)).reshape(-1, 20)
    for n in range(size):
        result_img[20 * n:20 * (n + 1), :] = img_list[n]
    return result_img


if __name__ == '__main__':
    knn, train, train_lable = initKnn()
    knn, train, train_lable = updateKnn(knn, train, train_lable)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        rois, th_img = findRois(frame)
        img_list = []
        for roi in rois:
            x, y, w, h = roi
            digit, img = findDigit(knn, th_img[y:y + h, x:x + w])
            img_list.append(img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
        help_img = cv2.cvtColor(th_img, cv2.COLOR_GRAY2BGR)
        new_frame = np.hstack((frame, help_img))
        cv2.imshow("frame", new_frame)
        key = cv2.waitKey(50) & 0xff
        if key == ord(" "):
            break
        elif key == ord("x"):
            digit_len = len(img_list)
            digit_img = concatenate(img_list)
            cv2.imshow("digits", digit_img)
            key = cv2.waitKey(0) & 0xff
            if key == ord("c"):
                data = input("请输入矫正的数据（空格区分）:")
                num_list = data.split(" ")
                print(num_list)
                if len(num_list) != digit_len:
                    print("训练失败，输入数据个数不对")
                    continue
                else:
                    try:
                        for i in range(digit_len):
                            num_list[i] = int(num_list[i])
                        knn, train, train_lable = updateKnn(knn, train, train_lable, digit_img, num_list)
                        print("训练成功!")
                    except Exception as e:
                        print(e)
                        print("训练失败，输入数据个数不是数字")
                        continue

    print("train len =", len(train))
    print("train_label len =", len(train_lable))
    # 将本次训练完的所有数据存储起来，方便下次使用
    np.savetxt("train.csv", train, fmt='%f', delimiter=',')
    np.savetxt("train_label.csv", train_lable, fmt='%d', delimiter=',')
    cap.release()  # 释放摄像头资源，关闭设备
    cv2.destroyAllWindows()  # 关闭所有窗口
