import cv2
import numpy as np

def nothing(x):  # 滑动条的回调函数
    pass


def main():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret,img = cap.read()
        skinMask = HSVBin(img)
        skinMask = opencloes(skinMask)

        #img = cv2.bilateralFilter(img, 5, 50, 100)  # 双边滤波
        #contours = getContours(skinMask)
        #trueContours = getTrueContours(skinMask)
        #hull = cv2.convexHull(trueContours, returnPoints=False)  # 计算轮廓的凸点
        #defects = cv2.convexityDefects(trueContours, hull)  # 计算轮廓的凹点
        #print(hull)-
        #print(defects)
        #print(len(contours))
        #print(type(contours))
        #print(type(contours[0]))
        #print(len(contours))


        #print(len(contours[0]))
        #contours, h = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.drawContours(img,contours,-1,(0,0,255),5)
        #cv2.drawContours(img,trueContours, -1, (0, 255, 0), 3)
        '''vaildContours = []
        for cont in contours:
            if cv2.contourArea(cont) > 9000:
                # x,y,w,h = cv2.boundingRect(cont)
                # if h/w >0.75:
                # filter face failed
                vaildContours.append(cv2.convexHull(cont))
                hull = cv2.convexHull(cont)

        cv2.drawContours(img, vaildContours, -1, (0, 255, 0), 5)
        res = cv2.bitwise_and(img, img, mask=skinMask)
        cv2.imshow('capture',img)
        cv2.imshow('55', res)


        #print(len(contours[0]))
        print(len(vaildContours[0]))
        contours[0]=[]'''

        # _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(skinMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 寻找轮廓   注：这里的'_'用作变量名称，_表示一个变量被指定了名称，但不打算使用。
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # 找到最大的轮廓（根据面积）
                temp = contours[i]
                area = cv2.contourArea(temp)  # 计算轮廓区域面积
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]  # 得出最大的轮廓区域
            hull = cv2.convexHull(res)  # 得出点集（组成轮廓的点）的凸包
            #defect = cv2.convexityDefects(res,hull)
            drawing = np.zeros(img.shape, np.uint8)



            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)  # 画出最大区域轮廓
            cv2.drawContours(img, [res], 0, (0, 255, 0), 2)  # 画出最大区域轮廓
            cv2.imshow('con', img)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)  # 画出凸包轮廓
            cv2.drawContours(img, [hull], 0, (0, 0, 255), 3)  # 画出凸包轮廓
            cv2.imshow('hull', img)
            moments = cv2.moments(res)  # 求最大区域轮廓的各阶矩
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            #moment.m00是零阶矩,可以用来表示图像的面积,moment.m10、moment.m01为一阶矩；Xc = m10/m00;Yc = m01/m00用来表示图像的重心.
            cv2.circle(drawing, center, 8, (0, 0, 255), -1)  # 画出重心
            cv2.circle(img, center, 8, (0, 0, 255), -1)  # 画出重心
            fingerRes = []  # 寻找指尖
            max = 0;
            count = 0;
            notice = 0;
            cnt = 0
            num = 0
            for i in range(len(res)):
                temp = res[i]
                dist = (temp[0][0] - center[0]) * (temp[0][0] - center[0]) + (temp[0][1] - center[1]) * (
                            temp[0][1] - center[1])  # 计算重心到轮廓边缘的距离
                if dist > max:
                    max = dist
                    notice = i

                if dist != max:
                    count = count + 1
                    if max < 25000:
                        continue
                    if count > 80:
                        count = 0
                        max = 0
                        flag = False  # 布尔值
                        if center[1] < res[notice][0][1]:  # 低于手心的点不算
                            continue
                        if dist < 900:
                            continue

                        for j in range(len(fingerRes)):  # 离得太近的不算
                            if abs(res[notice][0][0] - fingerRes[j][0]) < 40:
                                flag = True
                                break
                        for j in range(len(fingerRes)):  # 离得太远的不算
                            if abs(res[notice][0][0] - fingerRes[j][0]) > 500:
                                flag = True
                                break
                        if flag:
                            continue

                        fingerRes.append(res[notice][0])
                        cv2.circle(drawing, tuple(res[notice][0]), 8, (255, 0, 0), -1)  # 画出指尖
                        cv2.circle(img, tuple(res[notice][0]), 8, (255, 0, 0), -1)  # 画出指尖
                        #cv2.circle(img, tuple(defect), 8, (255, 0, 0), -1)  # 画出指尖

                        cv2.line(drawing, center, tuple(res[notice][0]), (255, 0, 0), 2)
                        cv2.line(img, center, tuple(res[notice][0]), (255, 0, 0), 2)

                        cnt = cnt + 1

            print("手指根数:")
            print(cnt)
            cv2.imshow('output', drawing)
            cv2.imshow('ori', img)
        k = cv2.waitKey(10)
        if k == 27:
            break



def getContours(img):
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    closed = cv2.morphologyEx(closed,cv2.MORPH_CLOSE,kernel)
    #contours,h = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    binary, contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    vaildContours = []
    for cont in contours:
        if cv2.contourArea(cont)>7000:
            #x,y,w,h = cv2.boundingRect(cont)
            #if h/w >0.75:
            #filter face failed
            vaildContours.append(cv2.convexHull(cont))

            #rect = cv2.minAreaRect(cont)
            #box = cv2.cv.BoxPoint(rect)
            #vaildContours.append(np.int0(box))
    return  vaildContours

def opencloes(img):
    kernel = np.ones((9, 9), np.uint8)
    #closed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closed', closed)
    return closed

def getTrueContours(img):
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    closed = cv2.morphologyEx(closed,cv2.MORPH_CLOSE,kernel)
    #contours,h = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours,h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vaildContours = []
    for cont in contours:
        if cv2.contourArea(cont) > 7000:
            # x,y,w,h = cv2.boundingRect(cont)
            # if h/w >0.75:
            # filter face failed
            vaildContours.append(cont)

            # rect = cv2.minAreaRect(cont)
            # box = cv2.cv.BoxPoint(rect)
            # vaildContours.append(np.int0(box))
    return vaildContours

    #return  contours

#输入BGR转换成HVS，提取感兴趣区域并二值化
def HSVBin(img):
    #hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_skin = np.array([100, 0, 0])
    upper_skin = np.array([125,255,255])

    '''cv2.namedWindow('i')
    cv2.createTrackbar('H', 'i', 0, 180, nothing)
    cv2.createTrackbar('S', 'i', 25, 255, nothing)
    cv2.createTrackbar('V', 'i', 0, 255, nothing)

    lowh = cv2.getTrackbarPos('H', 'i')
    lows = cv2.getTrackbarPos('S', 'i')
    lowv = cv2.getTrackbarPos('V', 'i')'''

    lower_skin = np.array([0, 25, 0])
    upper_skin = np.array([27, 100, 255])



    kernel = np.ones((15, 15), np.float32) / 225
    smoothed = cv2.filter2D(hsv, -1, kernel)

    median = cv2.medianBlur(hsv, 15)
    bilateral = cv2.bilateralFilter(img, 15, 75, 75)
    bilateral1 = cv2.bilateralFilter(hsv, 15, 75, 75)
    #lower_skin = np.array([100,25,0])
    #upper_skin = np.array([125,255,255])
    blur0 = cv2.GaussianBlur(hsv, (21, 21), 0)  # 加高斯模糊
    blur1 = cv2.GaussianBlur(img, (21, 21), 0)  # 加高斯模糊
    mask0 = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.inRange(blur0 ,lower_skin,upper_skin)
    mask1 = cv2.inRange(bilateral1,lower_skin,upper_skin)
    res = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('two1', mask1)
    cv2.imshow('two', mask)
    cv2.imshow('blur1', blur1)
    cv2.imshow('bilateral', bilateral)
    cv2.imshow('res', res)
    #blur = cv2.GaussianBlur(mask, (5, 5), 0)  # 加高斯模糊
    #cv2.imshow('blur', blur)

    #res = cv2.bitwise_and(img,img,mask=mask)
    return mask0
    #return blur

def HSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([9, 0, 46])
    upper_skin = np.array([27, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    res = cv2.bitwise_and(img, img, mask=mask)

    # mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    kernel = np.ones((15, 15), np.float32) / 225
    smoothed = cv2.filter2D(res, -1, kernel)
    cv2.imshow('Original', img)
    cv2.imshow('Averaging', smoothed)
    blur = cv2.GaussianBlur(res, (15, 15), 0)
    cv2.imshow('Gaussian Blurring', blur)
    median = cv2.medianBlur(res, 15)
    cv2.imshow('Median Blur', median)
    bilateral = cv2.bilateralFilter(res, 15, 75, 75)
    cv2.imshow('bilateral Blur', bilateral)


    # res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('res', res)
    return bilateral

def RGBHSV(R,G,B):
    max = max(R, G, B)
    min = min(R, G, B)
    V = max(R, G, B)
    S = (max - min) / max
    if (R == max):
        H =(G-B) / (max-min) * 60
    if (G == max):
        H = 120+(B-R) / (max-min) * 60
    if (B == max):
        H = 240 +(R-G) / (max-min) * 60
    if (H < 0):H = H+ 360

if __name__ =='__main__':
    main()
    