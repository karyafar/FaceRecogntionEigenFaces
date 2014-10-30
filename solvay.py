# A eigenfaces based face-recognition module.

#Orignially intended for work on intelligent recognition of Yale Face Database.


# It reads faces from the picture of Solvay Conference, studded with famous minds. 

#the path for solvay conference group picture
solvay = "Solvay/"
solvay_image = "Solvay/solvay2.jpg"

#it would be cool to use it on Manchester United squad as well.
manu = "ManUnited/"
manu_image = "ManUnited/manu.jpeg"

#default problem = solvay image
path = solvay
image_path  = solvay_image

#get imports
import cv2
import numpy as np


#return the picture by reading from the global path.
def getpicture():
    image = cv2.imread(image_path)
    return image

#The index.txt file contains the names of people in the picture
def gettotal():
    #get the total number of known faces.
    infile = open( path + "/index.txt")
    count = 0
    for line in infile:
        count = count +1
    return count


    
def save_face(img,name):
    #if a new face is detected, it should be saved for future recognition.
    count = gettotal()
    path_save = path + str(count+1) + ".PGM"
    out_file = open(path + "index.txt","a")
    out_file.write(path_save +";" + name+ "\n")
    out_file.close()
    cv2.imwrite(path_save,img)

def read_images(path, sz=None):
    names = []
    c = 0
    X,y = [], []
    
            #FILE FORMAT : PATH/ NAME
    
    
    infile = open(path+"index.txt")
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
    

def train():
    #train the model with EigenFaceRecognizer
    model = cv2.createEigenFaceRecognizer()
    X,y,names_of_images = read_images(path)
    model.train(np.asarray(X), np.asarray(y))
    #print("Training Finished")
    return model,names_of_images

def main():
    
    #train with known images
    model,names = train()
    #read the image for recognition
    image_main = cv2.imread(image_path)
    
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #convert the image to grayscale
    gray= cv2.cvtColor(image_main,cv2.COLOR_BGR2GRAY)
 
    #get all the faces
    faces = face_cascade.detectMultiScale(gray,1.2,1)

    # Number of detected faces : print (len(faces))
    for (x,y,w,h) in faces:
        
        image_show = getpicture()
	#get the image with just the face cropped        
	img2 = gray[y:y+h,x:x+w]
        
                
        img2 = cv2.resize(img2,(92,112))

	#recognize...
        [plabel,pconfidence] = model.predict(np.asarray(img2))
	#threshold...can be adjusted for accuracy..but yeah,900 works fine.
        if pconfidence<900:

            cv2.rectangle(image_show,(x,y),(x+w,y+h),(255,0,0),2)
            #draw a green rectangle over the face and write the name of the person next to it.
            cv2.putText(image_show,names[plabel].rstrip('\n'), (x,y),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
            cv2.imshow("Faces",image_show)
            
            print names[plabel]
            cv2.waitKey()
        else:
            #if a face is detected but not recognized, the name of the person should be provided. The program will then train itself to recognize the new face.
            cv2.rectangle(image_show,(x,y),(x+w,y+h),(255,0,0),2)
        
            cv2.putText(image_show,"???".rstrip('\n'), (x,y),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
            cv2.imshow("Faces",image_show)
            
            print names[plabel]
            cv2.waitKey()
            name = input("Enter the name")
            save_face(img2,name)

    main()       
        
if __name__ == "__main__":
    main()
