import streamlit as st 
import mediapipe as mp 
import cv2 
#checkboc for starting camera
start_cam=st.sidebar.checkbox("Test")

#for displaying The incoming feed
image_disp=st.empty()

#initialize the keypoint model
mp_drawing = mp.solutions.drawing_utils
mp_full = mp.solutions.holistic

#Initialize webcam mdoel
cap = cv2.VideoCapture(0)

#only if the camera start is selected
if start_cam:
    #use the holistic moddel for full body detection
    with mp_full.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5,upper_body_only=True) as holistic:
        while(1):

            #read the camera feed
            _, image = cap.read()

            #RGB>BGR
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            #Inference from keypoint model
            results = holistic.process(image)
            
            #If Left Hand is detected (right cuz mirror image LOL)
            if results.left_hand_landmarks or results.right_hand_landmarks or results.pose_landmarks:

               
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_full.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_full.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_full.UPPER_BODY_POSE_CONNECTIONS )

        
            #Display the camera feed
            image_disp.image(image,use_column_width=True)
else:
    cap.release()