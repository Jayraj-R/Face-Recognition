
#------------------------------------- Creating Dataset for the model -----------------------------------------------


import cv2
import os

cam = cv2.VideoCapture(0)   #accessing camera
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id(numerical) ')

print("\n [INFO] Initializing face capture....")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()   #taking image inputs
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #converts colored images to grayscale
    faces = face_detector.detectMultiScale(gray, 1.3, 5)    #storing grayscaled images(dimensions)

    for (x,y,w,h) in faces:     #four cordinates of the rectangle

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)    #displaying live camera feed as it captures

    number_of_images = 30
    k = cv2.waitKey(100) & 0xff 
    if k == 27:     #press ESC to exit
        break
    elif count >= number_of_images:        #if number of images reaches 30 stop reading
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program")
cam.release()   #closes the camera
cv2.destroyAllWindows()     #closes every window


