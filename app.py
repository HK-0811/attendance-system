import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from utils import load_and_encode_images,recognize_face,register
from lang_query import lang_response


st.set_page_config("Attendance System ✔️❌",page_icon='☑️')

if 'new_face_added' not in st.session_state:
    st.session_state.new_face_added= False

# Load known faces
known_encodings, known_names = load_and_encode_images(new_face_added=st.session_state.new_face_added)

st.session_state.new_face_added = False

st.title("Attendance System !!!")

# Start the webcam
run = st.checkbox('Capture Attendance')
run1 = st.checkbox('Register New User')
FRAME_WINDOW = st.image([])

# Initialize webcam capture
cap = cv2.VideoCapture(0)

if run:
    while True:
        ret, frame = cap.read()
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the frame
        recognize_face(frame,rgb_frame,known_encodings,known_names)

        FRAME_WINDOW.image(rgb_frame)


if run1:
    name = st.text_input("Enter Name")
    img_file_buffer = st.camera_input("Take Your Picture")
    if img_file_buffer is not None:

        # Register a new user
        register(name,img_file_buffer)
        st.session_state.new_face_added= True



user_query = st.chat_input("Ask..")
if user_query != "" and user_query is not None:

    attendance = pd.read_csv("attendance.csv")
    attendance_data = attendance.to_string(index=False) 

    st.chat_message("user").write(user_query)
    response = lang_response(user_query,attendance_data)
    st.chat_message("assistant").write(response)
