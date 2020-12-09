import cv2
import imutils
import time

# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture('color_2_without_skelet.mp4')

frame_rate = 1000
prev = 0


while True:
    time_elapsed = time.time() - prev
    res, image = cap.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()

        timer = cv2.getTickCount()
        ret, image = cap.read()
        if ret:
            image = imutils.resize(image,
                                   width=min(400, image.shape[1]))

            # Detecting all the regions
            # in the Image that has a
            # pedestrians inside it
            (regions, _) = hog.detectMultiScale(image,
                                                winStride=(4, 4),
                                                padding=(4, 4),
                                                scale=1.05)

            # Drawing the regions in the
            # Image
            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y),
                              (x + w, y + h),
                              (0, 0, 255), 2)

            # Showing the output Image
            fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
            cv2.putText(image, str(fps), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
            cv2.imshow("Image", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
