import cv2

imeg = cv2.imread("E:\\Data\\image\\[085657].jpg")
gry = cv2.cvtColor(imeg, cv2.COLOR_BGR2GRAY)
face = cv2.CascadeClassifier("E:\\Download\\haarcascade_frontalface_default.xml")
faces = face.detectMultiScale(gry, 19, 2)
for (x, y, w, h) in faces:
    imeg = cv2.rectangle(imeg, (x, y), (x + w, y + h), (127, 0, 205), 8)

imeg = cv2.resize(imeg, (900, 600))
imeg = cv2.imshow("Nazim", imeg)
cv2.waitKey()
cv2.destroyAllWindows()
