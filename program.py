# importing Libraries

import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime

# Get a reference to webcam

video_capture=cv2.VideoCapture(0)

# Load a sample picture and  recognize it.

vinayak_image=face_recognition.load_image_file("photos/vinayak.jpg")
vinayak_encoding=face_recognition.face_encodings(vinayak_image)[0]

steve_jobs_image=face_recognition.load_image_file("photos/stevejobs.jpg")
steve_jobs_encoding=face_recognition.face_encodings(steve_jobs_image)[0]

ratan_tata_image=face_recognition.load_image_file("photos/ratantata.jpg")
tatan_tata_encoding=face_recognition.face_encodings(ratan_tata_image)[0]

bill_gates_image=face_recognition.load_image_file("photos/billgates.jpg")
bill_gates_encoding=face_recognition.face_encodings(bill_gates_image)[0]

elon_musk_image=face_recognition.load_image_file("photos/elonmusk.jpg")
elon_musk_encoding=face_recognition.face_encodings(elon_musk_image)[0]

# Create arrays of known face encodings and their names

known_face_encodings = [
vinayak_encoding,    
steve_jobs_encoding,  
tatan_tata_encoding,
bill_gates_encoding,
elon_musk_encoding,  
]


known_face_names = [
"vinayak",
"steve jobs",
"ratan tata",
"bill gates",
"elon musk",
]

employees=known_face_names.copy()

# Initialize some variables

face_locations=[]
face_encodings=[]
face_name=[]
process=True

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(current_date+'.xlsx','w+')

lnwriter=csv.writer(f)

while True:
     # Grab a single frame of video
    ret,frame=video_capture.read()
    
    if process:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:,:,::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        
        face_names=[]
        for face_encoding in face_encodings:

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
            name="Unknown"
            face_distance=face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]

            face_names.append(name)
            # validate employees entry
            if name in employees:
                employees.remove(name)
                print(employees)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow({ name, current_time })

    process = not process

    #display the result
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
             
                    
    cv2.imshow("Attandance System",frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break  

video_capture.release()
cv2.destroyAllWindows()
f.close()

