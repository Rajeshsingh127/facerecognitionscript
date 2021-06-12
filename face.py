import cv2
import sys
imagepath = sys.argv[1]
cascpath = "haarcascade_frontalface_default.xml"

facecascade = cv2.CascadeClassifier(cascpath)


image = cv2.imread(imagepath)
gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = facecascade.detectMultiScale(
        gray,
        scaleFactor= 1.1,
        minNeighbors=5,
        minSize=(30,30),
        #flags=cv2.CV_HAAR_SCALE_IMAGE

)


print( "FOUND {0} faces!".format(len(faces)))

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("face found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

