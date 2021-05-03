from scipy.spatial import distance
import mediapipe as mp
from moviepy.editor import VideoFileClip, concatenate_videoclips
import json
import numpy as np
def give_cord_left(results,index):
    x = results.left_hand_landmarks.landmark[index].x
    y = results.left_hand_landmarks.landmark[index].y
    z = results.left_hand_landmarks.landmark[index].z
    return np.array([x,y,z])
def give_cord_right(results,index):
    x = results.right_hand_landmarks.landmark[index].x
    y = results.right_hand_landmarks.landmark[index].y
    z = results.right_hand_landmarks.landmark[index].z
    return np.array([x,y,z])


def give_vev(p1,p2):
    dist=distance.euclidean(p1,p2)
    return dist

def get_features_test(results,if_left,if_right):
    f={}
    if if_right==True:
        pt_lst_right = list(range(0,21))
        combo_right =[]
        for x in pt_lst_right:
            for y in pt_lst_right:
                if x!=y:
                    combo_right.append([x,y])
        
        for i,c in enumerate(combo_right):
            f[f'f{i}_right']=give_vev(give_cord_right(results,c[0]),give_cord_right(results,c[1]))
    elif if_right==False:
        pt_lst_right = list(range(0,21))
        combo_right =[]
        for x in pt_lst_right:
            for y in pt_lst_right:
                if x!=y:
                    combo_right.append([x,y])
        
        for i,c in enumerate(combo_right):
            f[f'f{i}_right']=0.0

    if if_left==True:
        pt_lst_left = list(range(0,21))
        combo_left =[]
        for x in pt_lst_left:
            for y in pt_lst_left:
                if x!=y:
                    combo_left.append([x,y])
        
        for i,c in enumerate(combo_left):
            f[f'f{i}_left']=give_vev(give_cord_left(results,c[0]),give_cord_left(results,c[1]))
    elif if_left==False:
        pt_lst_left = list(range(0,21))
        combo_left =[]
        for x in pt_lst_left:
            for y in pt_lst_left:
                if x!=y:
                    combo_left.append([x,y])
        
        for i,c in enumerate(combo_left):
            f[f'f{i}_left']=0.0
    
    return f


def get_features(results,class_,if_left,if_right):
    f={}
    if if_right==True:
        pt_lst_right = list(range(0,21))
        combo_right =[]
        for x in pt_lst_right:
            for y in pt_lst_right:
                if x!=y:
                    combo_right.append([x,y])
        
        for i,c in enumerate(combo_right):
            f[f'f{i}_right']=give_vev(give_cord_right(results,c[0]),give_cord_right(results,c[1]))
    elif if_right==False:
        pt_lst_right = list(range(0,21))
        combo_right =[]
        for x in pt_lst_right:
            for y in pt_lst_right:
                if x!=y:
                    combo_right.append([x,y])
        
        for i,c in enumerate(combo_right):
            f[f'f{i}_right']=0.0

    if if_left==True:
        pt_lst_left = list(range(0,21))
        combo_left =[]
        for x in pt_lst_left:
            for y in pt_lst_left:
                if x!=y:
                    combo_left.append([x,y])
        
        for i,c in enumerate(combo_left):
            f[f'f{i}_left']=give_vev(give_cord_left(results,c[0]),give_cord_left(results,c[1]))
    elif if_left==False:
        pt_lst_left = list(range(0,21))
        combo_left =[]
        for x in pt_lst_left:
            for y in pt_lst_left:
                if x!=y:
                    combo_left.append([x,y])
        
        for i,c in enumerate(combo_left):
            f[f'f{i}_left']=0.0
    
    f['class']=class_
    return f


def concat_gesture(g_list):
    clips=[]
    with open('clips.json','r') as f:
        clipmap = json.load(f)

    for g in g_list:
        clips.append(VideoFileClip(clipmap[g]))
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("output.mp4")
