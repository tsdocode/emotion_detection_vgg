from facenet_pytorch import MTCNN


import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
mtcnn = MTCNN()
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

def text(t, img, p):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = p
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2
    org                    = (0,100)

    cv2.putText(img,t, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

while True:
    # Read the frame
    _, img = cap.read()
    # Detect the faces
    face = img

    faces = mtcnn.detect(img)[0]
    # Draw the rectangle around each face
    if (faces != None):
        for (x, y, x1, y1) in faces:
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
            face = img[x:x1, y:y1]
   
    cv2.imshow('img', face) 
                                  

    # Display
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()