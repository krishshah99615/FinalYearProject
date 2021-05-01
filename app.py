import streamlit as st 
import mediapipe as mp
import pandas as pd
import os
from utils import give_cord,give_vev,get_features_test,get_features,concat_gesture
import cv2
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle as pkl 
import glob
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from gtts import gTTS
import playsound

MODEL_DIR='model'
GESTURE_DIR='gestures'
lang='en'

st.header('Gesture Tool')
st.text('Aiding disabled via AI')
nav_menu = ['Module1 (gesture2audio/text/gesture)','SelfTraining']
nav_select = st.sidebar.selectbox('Navigate',nav_menu)


if nav_select =='Module1 (gesture2audio/text/gesture)': 
    ouput_options = ['Audio','Animation']
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
        with mp_full.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
            while(1):

                #read the camera feed
                _, image = cap.read()

                #RGB>BGR
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                #Inference from keypoint model
                results = holistic.process(image)
                
                #If Left Hand is detected (right cuz mirror image LOL)
                if results.left_hand_landmarks:

                    #Get the features and reshape it for model infernce
                    f = get_features_test(results)
                    f = np.array(list(f.values())).reshape(1,-1)
 
                    #Inference and Print the label
                    p=model.predict(f)
                    token=str(label_enc.inverse_transform([p[0]])[0])
                    status.title(token)
                    
                    if token=="Start":
                        rec =True
                    
                    if token == "Stop":
                        rec = False
                        st.sidebar.text(" ".join(sent))
                        if ouput_select == 'Audio':
                            output=gTTS(" ".join(sent))
                            output.save("a.mp3")
                            playsound.playsound('a.mp3')
                        elif ouput_select == 'Animation':
                            sent.remove('Start')
                            #concat_gesture(['Hello','Happy'])
                            concat_gesture(sent)
                            st.video('output.mp4')
                        
                        sent = []
                    if rec:
                        if last_label !=token:
                            sent.append(token)
                            last_label=token
                    



                else:
                    #if no hand is detected
                    status.title(None)
                #Draw the mesh and keyponts
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_full.HAND_CONNECTIONS)
            
                #Display the camera feed
                image_disp.image(image,use_column_width=True)
    else:
        cap.release()








elif nav_select == 'SelfTraining':
    train_menu = ['Collecting','Training','Testing']
    train_select=st.sidebar.selectbox('Model Making Phase',train_menu)

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
            with mp_full.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
                while(1):

                    #read the camera feed
                    _, image = cap.read()

                    #RGB>BGR
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                    #Inference from keypoint model
                    results = holistic.process(image)
                    
                    #If Left Hand is detected (right cuz mirror image LOL)
                    if results.left_hand_landmarks:

                        #Get the features and reshape it for model infernce
                        f = get_features_test(results)
                        f = np.array(list(f.values())).reshape(1,-1)

                        #Inference and Print the label
                        p=model.predict(f)
                        status.title(str(label_enc.inverse_transform([p[0]])[0]))
                    else:
                        #if no hand is detected
                        status.title(None)
                    #Draw the mesh and keyponts
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_full.HAND_CONNECTIONS)
                
                    #Display the camera feed
                    image_disp.image(image,use_column_width=True)
        else:
            cap.release()

    elif train_select == 'Training':
        status = st.empty()

        df=pd.DataFrame()
        for json_file_loc in glob.glob(f"{GESTURE_DIR}/*.json"):
            df = df.append(pd.read_json(json_file_loc))

        st.subheader('Various Gesture Info')
        st.table(df['class'].value_counts())

        st.subheader('Data')
        st.table(df.head())

        
        label_encoder = preprocessing.LabelEncoder() 
        df['class']= label_encoder.fit_transform(df['class'])

        test_ratio=st.sidebar.slider("Test Ratio",0.1,0.9,0.3,0.1,"%f%%")
        layers = int(st.sidebar.slider("Layers",1,5,3,1))
        neurons = int(st.sidebar.slider("Neurons",10,50,30,5))
        layout=[]
        for layer in range(layers):
            layout.append(neurons)
        mlp = MLPClassifier(hidden_layer_sizes=tuple(layout))

        X = df.drop('class',axis=1)
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=float(test_ratio))

        start_train=st.sidebar.checkbox("Start Training")
        if start_train:
            mlp.fit(X_train,y_train)
            
            with open(f'{MODEL_DIR}/model.pkl', 'wb') as fh:
                pkl.dump(mlp, fh)
            with open('labelenc.pkl', 'wb') as fh:
                pkl.dump(label_encoder, fh)
            status.success("Model Saved at "+f'{MODEL_DIR}/model.pkl')
        
            predictions = mlp.predict(X_test)
            acc = accuracy_score(y_test,predictions)
            st.subheader("Test Accuracy :"+str(acc*100)+"%")

            st.subheader('Loss Graph')
            st.line_chart(mlp.loss_curve_)
            

            cm=confusion_matrix(y_test,predictions)

            fig, ax = plt.subplots(figsize=(3,3))
            sns.heatmap(cm,cbar=False,annot=True,xticklabels=list(label_encoder.classes_),yticklabels=list(label_encoder.classes_))
            st.write(fig)

    elif train_select=='Collecting':

        #Initialize empty list for json
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
            with mp_full.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
                while start_cam:
                    

                    #reading webcam
                    _, image = cap.read()

                    #BGR>RGB
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                    #model inference for keypoint detection
                    results = holistic.process(image)

                    # Code might break incase hand goes out of frame 
                    try:

                        #Drawuing points and mesh on the image
                        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_full.HAND_CONNECTIONS)
                        
                        #run only if collect data is selected
                        if col:
                            #print the counter value
                            collected.text("Collected : "+str(counter))
                            
                            #Get 8 features[Mnually set] and append in the json  
                            data.append(get_features(results,name))
                            
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
