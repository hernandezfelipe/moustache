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
mouthCascade = cv2.CascadeClassifier(path+"\haar_mouth.xml")
noseCascade = cv2.CascadeClassifier(path+"\haar_nose.xml")
eyecascade = cv2.CascadeClassifier(path+'\haar_eye.xml')
moust = cv2.imread(path+"\moustache.png", -1)
# cross = cv2.imread(path+"\cross.png", -1)

while True:

    try:
        s, image  = cam.read()	
        img = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)


        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            roi_gray = gray[y + int(h/2) : y + h, x:x+int(w)]
            roi_color = image[y + int(h/2) : y + h, x:x+int(w)] 
            
#            roi_gray_eye = gray[y : y + int(h/2), x:x+int(w)]
#            roi_color_eye = image[y : y + int(h/2), x:x+int(w)]  
            
           
            
#            eyes = eyecascade.detectMultiScale(roi_gray_eye)
#            mouths = mouthCascade.detectMultiScale(roi_gray)
#            noses = noseCascade.detectMultiScale(roi_gray)

#            for (ex,ey,ew,eh) in eyes:
#                 cv2.rectangle(roi_color_eye,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
               
#            for (nx,ny,nw,nh) in noses:
#                 cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,255,0),2)
#                
#            for (mx,my,mw,mh) in mouths:
#            
#                cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)
##                     
        moustache = cv2.resize(moust, (w, int(w/2)))         
        x_offset, y_offset = x + w/2 - moustache.shape[1]/2 , y + h/2 + moustache.shape[0]/16
        y1, y2 = y_offset, y_offset + moustache.shape[0]
        x1, x2 = x_offset, x_offset + moustache.shape[1]
        alpha_s = moustache[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
   
        for c in range(3):
           
           image[y1:y2, x1:x2, c] = (alpha_s * moustache[:, :, c] + alpha_l * image[y1:y2, x1:x2, c])
            
              
        # x_offset, y_offset = x + w/2 - cross.shape[1]/2 , y + h/2 - cross.shape[0]/2 
        # y1, y2 = y_offset, y_offset + cross.shape[0]
        # x1, x2 = x_offset, x_offset + cross.shape[1]
        # alpha_s = cross[:, :, 3] / 255.0
        # alpha_l = 1.0 - alpha_s
    
        # for c in range(3):
            
            # image[y1:y2, x1:x2, c] = (alpha_s * cross[:, :, c] + alpha_l * image[y1:y2, x1:x2, c])       
        
            
            
        cv2.imshow("faces", image)
        cv2.waitKey(1) 
        
    except KeyboardInterrupt:
        cam.release()
    except:
        cv2.imshow("faces", img)
        cv2.waitKey(1) 
