from scipy.spatial import distance
import mediapipe as mp
import numpy as np
def give_cord(results,index):
    x = results.left_hand_landmarks.landmark[index].x
    y = results.left_hand_landmarks.landmark[index].y
    z = results.left_hand_landmarks.landmark[index].z
    return np.array([x,y,z])
def give_vev(p1,p2):
    dist=distance.euclidean(p1,p2)
    return dist
def get_features_test(results):
    pt_lst = list(range(0,21))
    combo =[]
    for x in pt_lst:
        for y in pt_lst:
            if x!=y:
                combo.append([x,y])
    f={}
    for i,c in enumerate(combo):
        f[f'f{i}']=give_vev(give_cord(results,c[0]),give_cord(results,c[1]))
 
    return f
def get_features(results,class_):
    pt_lst = list(range(0,21))
    combo =[]
    for x in pt_lst:
        for y in pt_lst:
            if x!=y:
                combo.append([x,y])
    f={}
    for i,c in enumerate(combo):
        f[f'f{i}']=give_vev(give_cord(results,c[0]),give_cord(results,c[1]))
    
    f['class']=class_
    return f
    