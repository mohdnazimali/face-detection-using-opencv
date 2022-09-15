import cv2
#this path is my local disk path image 
image = cv2.imread("E:\\Data\\image\\[085657].jpg")

gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#this data det also in my local disk path 
face = cv2.CascadeClassifier("E:\\Download\\haarcascade_frontalface_default.xml")
faces = face.detectMultiScale(gry, 19, 2)
for (x, y, w, h) in faces:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 205), 8)

image = cv2.resize(image, (900, 600))
image = cv2.imshow("Nazim", image)
cv2.waitKey()
cv2.destroyAllWindows()
