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
    return{
        'f1':give_vev(give_cord(results,1),give_cord(results,17)),
        'f2':give_vev(give_cord(results,2),give_cord(results,13)),
        'f3':give_vev(give_cord(results,3),give_cord(results,9)),
        'f4':give_vev(give_cord(results,4),give_cord(results,5)),
        'f5':give_vev(give_cord(results,0),give_cord(results,4)),
        'f6':give_vev(give_cord(results,0),give_cord(results,8)),
        'f7':give_vev(give_cord(results,0),give_cord(results,12)),
        'f8':give_vev(give_cord(results,0),give_cord(results,16)),
    }
def get_features(results,class_):
    return{
        'f1':give_vev(give_cord(results,1),give_cord(results,17)),
        'f2':give_vev(give_cord(results,2),give_cord(results,13)),
        'f3':give_vev(give_cord(results,3),give_cord(results,9)),
        'f4':give_vev(give_cord(results,4),give_cord(results,5)),
        'f5':give_vev(give_cord(results,0),give_cord(results,4)),
        'f6':give_vev(give_cord(results,0),give_cord(results,8)),
        'f7':give_vev(give_cord(results,0),give_cord(results,12)),
        'f8':give_vev(give_cord(results,0),give_cord(results,16)),
        "class":class_

    }