#[Navin_Kumar_Manaswi]_Deep_Learning_with_Applicati(z-lib.org).pdf
# Multi face tracking :  https://www.guidodiepen.nl/2017/02/tracking-multiple-faces/

import cv2
import dlib
import os

saved_faces=[]
save_path=os.path.join(os.getcwd(),"Faces")
facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eyecascade=cv2.CascadeClassifier('haarcascade_eye.xml')

path='D:/projects/ATM public safety/Third Review/Face detection and MultiFace Tracking/Extraphotos'
for filename in os.listdir(path):
    img=cv2.imread(os.path.join(path,filename))
    #vs=cv2.VideoCapture(0)
    face_count=0
    eye_count=0
    orig_img=img.copy()
        
    gray_img=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)

        
    faces=facecascade.detectMultiScale(gray_img,1.3,5)
    #eyes=eyecascade.detectMultiScale(gray_img,1.3,5)

    x=0
    y=0
    w=0
    h=0
    max_area=0

    for (_x,_y,_w,_h) in faces:
        if _w*_h>max_area:
            x=_x
            y=_y
            h=_h
            w=_w
            max_area=_w*_h
            print("writing face")
            cv2.imwrite(os.path.join(save_path,'{0}.jpg'.format(len(os.listdir(save_path))+1)),img[ y-20:y+h+20,x-10:x+w+10])
        
