import cv2

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

imgNum  = 0
for i in range(1,3):
  img = cv2.imread('./train/img'+str(i)+'.jpg')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3,5)
  for (x,y,w,h) in faces:
    cropped = img[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
    cv2.imwrite("./result/cropped" + str(imgNum) + ".jpg", cropped)
  imgNum += 1
