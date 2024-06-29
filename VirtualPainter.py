import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
#################
brushThickness  = 15
eraserThickness = 100
#################
save_key = ord('s')  # Phím 's' để lưu hình ảnh

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(min_detection_confidence=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# # Thời gian bắt đầu
# start_time = time.time()
# run_time = 10 # Thời gian chạy (giây)

save_images = False

while True:
    #1. Import  image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #2. Find hand landmarks
    img = detector.find_hands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:

        #print(lmList)
        #tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        #4. If selection mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            #Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)




        #5. If Drawing mode - index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img,  (xp, yp),  (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas,  (xp, yp),  (x1, y1), drawColor, brushThickness)


            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_BGRA2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    #Seting the header image
    img[0:125, 0:1280] = header

    # # Tính thời gian đã trôi qua
    # elapsed_time = time.time() - start_time
    #
    # # Hiển thị thời gian lên hình ảnh
    # cv2.putText(img, f"Time Elapsed: {int(elapsed_time)} seconds", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 0, 255), 2)

    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)

    # Kiểm tra xem có muốn lưu hình ảnh không
    if save_images:
        # Lưu hình ảnh
        cv2.imwrite("image.png", img)
        print("Image saved!")


    # Kiểm tra xem phím tắt đã được nhấn chưa
    key = cv2.waitKey(1)
    if key == save_key:
        save_images = not save_images # Kích hoạt lưu hình ảnh



    # # Kiểm tra xem đã đạt đến thời gian chạy tối đa chưa
    # if elapsed_time >= run_time:
    #     # Lưu hình ảnh
    #     cv2.imwrite("final_drawing.jpg", img)
    #     break

 # Giải phóng tài nguyên và đóng cửa sổ
# cap.release()
# cv2.destroyAllWindows()

