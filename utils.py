import face_recognition
import pandas as pd
from datetime import datetime
import os
import streamlit as st
import cv2
import numpy as np

@st.cache_resource
def load_and_encode_images(directory='people_db',new_face_added=True):
    encodings = []
    names = []

    for filename in os.listdir(directory):
        if filename.endswith((".jpg",".jpeg",".png")):
            image_path = os.path.join(directory,filename)
            image = face_recognition.load_image_file(image_path)
            try:
                encoding = face_recognition.face_encodings(image)[0]
            except IndexError:
                print(f"No face found for {image_path}.Please try capturing the image again")
            encodings.append(encoding)
            names.append(image_path.split('\\')[-1].split('.')[0])
            print(names)
    return encodings, names


def recognize_face(frame,rgb_frame,known_encodings,known_names):
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            mark_attendance(name)
        # Display name on the frame
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def initialize_csv(file_path="attendance.csv"):
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        attendance_df = pd.DataFrame(columns=["Name", "Date", "Status"])
        attendance_df.to_csv(file_path,index=False)


def mark_attendance(name, file_path="attendance.csv"):
    initialize_csv(file_path)
    attendance_df = pd.read_csv(file_path)

    current_date = datetime.now().strftime("%Y-%m-%d")
    if any((attendance_df['Name'] == name) & (attendance_df['Date'].str.startswith(current_date))):
        return
    
    attendance_record = {
            "Name": name,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Status": "Present"
        }

    attendance_df =  pd.concat([attendance_df, pd.DataFrame([attendance_record])], ignore_index=True)
    attendance_df.to_csv(file_path,index=False)


def register(name,img_file_buffer):
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    folder = "people_db"
    image_path = os.path.join(folder,f"{name}.jpg")
    cv2.imwrite(image_path,cv2_img)
    st.success(f"{name} Successfully Registered!")
    