import numpy as np
import os
try:
    import cv2
except:
    os.system('pip install opencv-python')
    import cv2
    
path = os.path.dirname(os.path.abspath(__file__))

cam = cv2.VideoCapture(0)   # 0 -> index of camera

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(path+"\haar_face.xml")
# mouthCascade = cv2.CascadeClassifier(path+"\haar_mouth.xml")
# noseCascade = cv2.CascadeClassifier(path+"\haar_nose.xml")
# eyecascade = cv2.CascadeClassifier(path+'\haar_eye.xml')
moustache = cv2.imread(path+"\moustache.png", -1)

while True:

    s, image  = cam.read()	
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
#        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y + int(h/2) : y + h, x:x+int(w)]
        roi_color = image[y + int(h/2) : y + h, x:x+int(w)]        
        moustache = cv2.resize(moustache, (100, 50))         
        x_offset, y_offset = x + w/2 - moustache.shape[1]/2 , y + h/w + h/5 + moustache.shape[0]    
        y1, y2 = y_offset, y_offset + moustache.shape[0]
        x1, x2 = x_offset, x_offset + moustache.shape[1]
        alpha_s = moustache[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
    
        for c in range(3):
            
            image[y1:y2, x1:x2, c] = (alpha_s * moustache[:, :, c] + alpha_l * image[y1:y2, x1:x2, c])
        
#        eyes = eyecascade.detectMultiScale(roi_gray)
#        mouths = mouthCascade.detectMultiScale(roi_gray)
#        noses = noseCascade.detectMultiScale(roi_gray)

        # for (ex,ey,ew,eh) in eyes:
			# cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
           
        # for (nx,ny,nw,nh) in noses:
			# cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
#            
#        for (mx,my,mw,mh) in mouths:
#        
#            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)
                            
		
		
    cv2.imshow("faces", image)
    cv2.waitKey(1) 