"""Face Dataset collection Using Libraries"""
import cv2, os

alg = "haarcascade_frontalface_default.xml"
haar_casecade = cv2.CascadeClassifier(alg)
datasets = "dataset"
subdata = input("Enter your name: \n\t")
cam = cv2.VideoCapture(0)

path =  os.path.join(datasets, subdata)
if not os.path.isdir(path):
    os.mkdir(path)
(width, height) = (130, 100)

count = 1
while count < 50:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar_casecade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path,count), face_resize)
        count += 1
    cv2.imshow("Collecing Data", img)
    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()
