import streamlit as st   # For Gui
import mediapipe as mp   # For Keypoint Detection
import pandas as pd      # For Data Manipulation
import os                # Operating Sys Operations
from utils import give_vev,get_features_test,get_features,concat_gesture    #Custom Fucntions
import cv2                # Image manipulation and  webcam 
import json               # Json processing
import numpy as np        # numerical processing
import pickle as pkl       # Storing models n label encoder
import glob                # Finding perticular extention file

from sklearn import preprocessing 
from sklearn.neural_network import MLPClassifier # Nueral Network
from sklearn.model_selection import train_test_split # Preprocessing steps
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score # Evaluation metrics

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Audio Libraires
from gtts import gTTS
import playsound


import librosa
import streamlit as st
from helper import create_spectrogram, read_audio, record, save_record
import speech_recognition as sr

# Setting the paths
MODEL_DIR='model'
GESTURE_DIR='gestures'
lang='en'

# basic html tags
st.header('Gesture Tool')
st.text('Aiding disabled via AI')

# different navigations pages
nav_menu = ['gesture2audio','text/audio2gesture','SelfTraining']

nav_select = st.sidebar.selectbox('Navigate',nav_menu)


#Module 2:
if nav_select=='text/audio2gesture':
    #Select input mode
    
    input_options = ['Audio','Text']
    input_select = st.sidebar.radio("Audio Output",input_options)
    #iniiltialse empty sentence input
    inp = ''
    
    if input_select == 'Text':
        #If input is text
        inp = st.sidebar.text_input("Sentence")
    if input_select == 'Audio':
        
        st.header("Record your own voice")
        ##reecord voice
        if st.button(f"Click to Record"):
            record_state = st.text("Recording...")
            duration = 5  # seconds
            fs = 48000
            filename='input'
            myrecording = record(duration, fs)
            record_state.text(f"Saving sample as {filename}.mp3")

            path_myrecording = f"{filename}.wav"

            save_record(path_myrecording, myrecording, fs)
            record_state.text(f"Done! Saved sample as {filename}.mp3")

            st.audio(read_audio(path_myrecording))
            r = sr.Recognizer()
            audio_inp = sr.AudioFile('input.wav')
            with audio_inp as source:
                audio = r.record(source)
                a= r.recognize_google(audio)
                st.text(a)
                inp = a
    if inp!='':
        clipsmap = open('clips.json','r')
        f=json.load(clipsmap)
        all_keys = [x.lower() for x in list(dict(f).keys())]
        tokens=[x.lower() for x in inp.split()]
        new_sent=[]
        for token in tokens:
            if token in all_keys:
                new_sent.append(token)
        if len(new_sent)>0:
            concat_gesture([x.capitalize() for x in new_sent])
            st.video('output.mp4')
        else:
            st.sidebar.error("Not avaible in our dictonary")


# Module 1 
elif nav_select =='gesture2audio': 
    
    # Set the ouput options apart form text
    ouput_options = ['Audio with Text','Only Text']
    ouput_select = st.sidebar.radio("Audio Output",ouput_options)

    #Cache the model for testing 
    @st.cache()
    def load_model():
        return pkl.load(open(MODEL_DIR+"/model.pkl",'rb'))
    model = load_model()

    #Load Label Encoder
    label_enc=pkl.load(open("labelenc.pkl",'rb'))
    
    

    #To print the Label detected
    status = st.empty()

    #checkboc for starting camera
    start_cam=st.sidebar.checkbox("Start Camera")

    #sentece list , recording boolean , last label predicted
    sent , rec ,last_label =[],False,''

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
                if_left=False
                if_right=False
                #If Left Hand is detected (right cuz mirror image LOL)
                if results.left_hand_landmarks or results.right_hand_landmarks:

                    #Get the features and reshape it for model infernce
                    if results.left_hand_landmarks:
                        if_left=True
                    if results.right_hand_landmarks:
                        if_right=True
                    
                    f = get_features_test(results,if_left,if_right)
                    f = np.array(list(f.values())).reshape(1,-1)
 
                    #Inference and Print the label
                    p=model.predict(f)
                    token=str(label_enc.inverse_transform([p[0]])[0])
                    status.title(token)
                    
                    # If start symbol
                    if token=="Start":
                        #recording start
                        rec =True
                    
                    # If stop symbol reset the sentence 
                    if token == "Stop":
                        rec = False
                        sent = [x for x in sent if x!='Start']
                        st.sidebar.text(" ".join(sent))
                        
                        #Play audio
                        if ouput_select == 'Audio with Text':
                            output=gTTS(" ".join(sent))
                            output.save("a.mp3")
                            playsound.playsound('a.mp3')
                        
                        
                        
                        sent = []
                    # To ensure labels dont repeat concecutively
                    if rec:
                        if last_label !=token:
                            sent.append(token)
                            last_label=token
                    



                else:
                    #if no hand is detected
                    status.title(None)
                #Draw the mesh and keyponts
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_full.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_full.HAND_CONNECTIONS)
            
                #Display the camera feed
                image_disp.image(image,use_column_width=True)
    else:
        cap.release()







# Module 3
elif nav_select == 'SelfTraining':
    # Model building steps navigation 
    # Collect data > Train model n Evaluate > Test Model 
    train_menu = ['Collecting','Training','Testing']
    train_select=st.sidebar.selectbox('Model Making Phase',train_menu)
    
    # Testing Phase
    if train_select == 'Testing':
        #Cache the model for testing 
        @st.cache()
        def load_model():
            return pkl.load(open(MODEL_DIR+"/model.pkl",'rb'))
        model = load_model()

        #Load Label Encoder
        label_enc=pkl.load(open("labelenc.pkl",'rb'))
        
        

        #To print the Label detected
        status = st.empty()

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
                    if_left=False
                    if_right=False
                    #If Left Hand is detected (right cuz mirror image LOL)
                    if results.left_hand_landmarks or results.right_hand_landmarks:

                        #Get the features and reshape it for model infernce
                        if results.left_hand_landmarks:
                            if_left=True
                        if results.right_hand_landmarks:
                            if_right=True
                        
                        f = get_features_test(results,if_left,if_right)
                        f = np.array(list(f.values())).reshape(1,-1)

                        #Inference and Print the label
                        p=model.predict(f)
                        status.title(str(label_enc.inverse_transform([p[0]])[0]))
                    else:
                        #if no hand is detected
                        status.title(None)
                    #Draw the mesh and keyponts
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_full.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_full.HAND_CONNECTIONS)
                
                    #Display the camera feed
                    image_disp.image(image,use_column_width=True)
        else:
            cap.release()
    # Training Phase
    elif train_select == 'Training':
        status = st.empty()
        
        #Load all json features stored
        df=pd.DataFrame()
        for json_file_loc in glob.glob(f"{GESTURE_DIR}/*.json"):
            df = df.append(pd.read_json(json_file_loc))

        st.subheader('Various Gesture Info')
        st.table(df['class'].value_counts())

        st.subheader('Data')
        st.table(df.head())

        
        label_encoder = preprocessing.LabelEncoder() 
        df['class']= label_encoder.fit_transform(df['class'])
        
        
        # Custom Trainig Neura Network
        test_ratio=st.sidebar.slider("Test Ratio",0.1,0.9,0.3,0.1,"%f%%")
        layers = int(st.sidebar.slider("Layers",1,5,3,1))
        neurons = int(st.sidebar.slider("Neurons",10,50,30,5))
        layout=[]
        for layer in range(layers):
            layout.append(neurons)
        
        mlp = MLPClassifier(hidden_layer_sizes=tuple(layout))
        
        # Preprocsiing
        X = df.drop('class',axis=1)
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=float(test_ratio))

        #start trianing 
        start_train=st.sidebar.checkbox("Start Training")
        if start_train:
            #Train
            mlp.fit(X_train,y_train)
            
            # Save model
            with open(f'{MODEL_DIR}/model.pkl', 'wb') as fh:
                pkl.dump(mlp, fh)
            # Save Label encoder
            with open('labelenc.pkl', 'wb') as fh:
                pkl.dump(label_encoder, fh)
            status.success("Model Saved at "+f'{MODEL_DIR}/model.pkl')
            # Predict for evaluation
            predictions = mlp.predict(X_test)
            acc = accuracy_score(y_test,predictions)
            st.subheader("Test Accuracy :"+str(acc*100)+"%")

            st.subheader('Loss Graph')
            st.line_chart(mlp.loss_curve_)
            

            cm=confusion_matrix(y_test,predictions)
            cr=classification_report(y_test,predictions)

            fig, ax = plt.subplots(figsize=(3,3))
            st.write(cr)
            sns.heatmap(cm,cbar=False,annot=True,xticklabels=list(label_encoder.classes_),yticklabels=list(label_encoder.classes_))
            st.write(fig)
    
    # Data Collection Phase
    elif train_select=='Collecting':

        #Initialize  empty list for json
        data=[]
        #Gesture Parameters
        st.sidebar.header("Collecting Train Data")
        st.sidebar.text("Enter The Gesture Details Below")
        name=st.sidebar.text_input("Name")
        #max Number of datapoint for a gesture for training
        n=int(st.sidebar.slider("No Of Samples",min_value=1,max_value=1000,step=100,value=200))

        #run Only if person enters label for the gesture
        if name:
            # Set Frame counter to 0 
            counter=0

            #Text area for printing Success at end of collecting
            status = st.empty()

            #checkbox for Starting camera
            start_cam=st.sidebar.checkbox("Start Camera")

            #Text area for display of counter and image
            collected = st.empty()
            image_disp=st.empty()

            #initializing keypoint detector model
            mp_drawing = mp.solutions.drawing_utils
            mp_full = mp.solutions.holistic

            #Webcam Capture object
            cap = cv2.VideoCapture(0)
            
            #checkbox to start collecting
            col = st.sidebar.checkbox("Colllect Gesture Data")

            #using the holisting full body model
            with mp_full.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5,upper_body_only=True) as holistic:
                while start_cam:
                    

                    #reading webcam
                    _, image = cap.read()

                    #BGR>RGB
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                    #model inference for keypoint detection
                    results = holistic.process(image)
                    if_left=False
                    if_right=False
                  
                    # Code might break incase hand goes out of frame 
                    try:

                        #Drawuing points and mesh on the image
                        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_full.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_full.HAND_CONNECTIONS)
                       
                        #run only if collect data is selected
                        if col:
                            #print the counter value
                            collected.text("Collected : "+str(counter))
                            
                            #Get 8 features[Mnually set] and append in the json 
                            if results.left_hand_landmarks:
                                if_left=True
                            if results.right_hand_landmarks:
                                if_right=True
                            
                            data.append(get_features(results,name,if_left,if_right))
                            
                            #increment the collection counter
                            counter=counter+1

                            #if max num of frames reached break out of the loop
                            if counter>n:
                                
                                #save the json data
                                with open(f"{GESTURE_DIR}/data-{name}.json",'w') as f:
                                    json.dump(data,f)
                                
                                #pritn the status
                                status.success("Collectd "+str(n)+" Datapoints")
                                
                                #reset counter
                                counter = 0
                                break
                    except AttributeError as a:    
                        print(a)
                    #Display the camera input
                    image_disp.image(image,use_column_width=True)
                    
 
