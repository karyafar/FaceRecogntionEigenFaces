import os
import sys
import cv2
import numpy as np
total = 0
model = cv2.createEigenFaceRecognizer()
names = []
detected = 0

#face detection and recognition on a video feed.


def gettotal():
    infile = open("pradeep_collection/index.txt")
    count = 0
    for line in infile:
        count = count +1
    return count
def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)



def read_images(path, sz=None):
    names = []
    """Reads the images in a given folder, resizes images on the fly if size is given.
Returns:
A list [X,y]
X: The images, which is a Python list of numpy arrays.
y: The corresponding labels (the unique number of the subject, person) in a Python list.
"""
    c = 0
    X,y = [], []
    
            #FILE FORMAT : PATH/ NAME
    
    
    infile = open("pradeep_collection/index.txt")
    count = 0
    for line in infile:
        data = line.split(";")
        #print data
        im = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)
        
        
         # resize to given size (if given)
        X.append(np.asarray(im, dtype=np.uint8))
        y.append(count)
        names.append(data[1])
        count = count+1
    total = count
    infile.close()
    return X,y,names
    

global_frame = None

out_dir = "pradeep_collection/"

def detect_face(img):
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray,1.5,5)
    #print len(faces)
    if len(faces)!=0:
        #print "Detected:",detected
        max = 0
        x1=0
        y1 = 0
        w1 = 0
        h1 = 0
        for (x,y,w,h) in faces:
            if h>max:
                max = h
                x1 = x
                y1 = y
                w1 = w
                h1 = h
            
        
        y1 = int(y1 - 0.1*h)
        x1 = int(x1 - 0.1*w)
        h1 = int(1.2*h1)
        w1 = int(1.2*w1)
        img2 = img[y1:y1+h1,x1:x1+w1]
        if y1<0 or x1<0:
            return [0,0,0,0,0,0,0]
        if y1>480 and x1>640:
            return [0,0,0,0,0,0,0]
        
        return [img2,1,gray,x1,y1,h1,w1]
    else:
        return [0,0,0,0,0,0,0]
    

def save_face(img,name):
    count = gettotal()
    path = out_dir + str(count+1) + ".PGM"
    out_file = open("pradeep_collection/index.txt","a")
    out_file.write(path +";" + name+ "\n")
    out_file.close()
    cv2.imwrite(path,img)
    
model = cv2.createEigenFaceRecognizer()


def load_faces():
    [X,y] = read_images(out_dir)
    model = cv2.createEigenFaceRecognizer()
    # Read
    # Learn the model. 
    model.train(np.asarray(X), np.asarray(y))
    
                        
def train():
    
    X,y,names_of_images = read_images(out_dir)
    model.train(np.asarray(X), np.asarray(y))
    #print("Training Finished")
    return model,names_of_images
    
cap = cv2.VideoCapture(0)







def main():
    
    total = 0
    model,names = train()
    while(1):
        detected = 0
    
    # read the frames
        _,frame = cap.read()
        #barney = cv2.imread("barney.jpg")
    # smooth it
        frame = cv2.blur(frame,(3,3))
        print len(frame),len(frame[0])
        frame_p = frame.copy()
        global_frame = frame_p.copy()
        img,det,full,x,y,h,w = detect_face(frame_p)
        
        cv2.imshow("WebCam feed",global_frame)
        k = cv2.waitKey(1)
        if det == 1:
            img2 = cv2.resize(img,(92,112))
            
            
            
            img2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            [plabel,pconfidence] = model.predict(np.asarray(img2))
            print pconfidence
        #TODO print the name beside the picture
            if gettotal == 0:
                threshhold = 1
            else:
                threshhold = 3000
            if pconfidence < threshhold :
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                #print len(names),plabel
                cv2.putText(frame,names[plabel].rstrip('\n'), (x,y),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
                cv2.imshow("Face",frame)
                
                #print names[plabel]
                cv2.waitKey(2)
                
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.imshow("Face",frame)
            
                cv2.waitKey(2)
                if k == 97:
                    name = input("Enter a name:?")
                    if (name!=""):
                        save_face(img2,name)
                        model,names = train()
                if k == 100:
                    main()
               
    
            
if __name__ == "__main__":
    total = 0
    main()
