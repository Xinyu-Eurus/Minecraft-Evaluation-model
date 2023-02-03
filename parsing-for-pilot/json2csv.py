import pandas as pd
import csv
import os
import sys
import json

import numpy as np
import sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

eva_str2int={"junior":1, "medium":2, "advanced":3}

#read survey files
def read_json(jsonPath): 
    jDatas=[]
    with open(jsonPath,'r') as f: 
        for line in f:
            try:
                jdata=json.loads(line.rstrip(';\n'))
#                 print(jdata)
                jDatas.append(jdata)
            except ValueError:
                print ("Skipping invalid line {0}".format(repr(line)))                    
    return jDatas

def match_m_ai(m_jdata,ai_jDatas):
    for jdata in ai_jDatas:
        if jdata["player"]=="worker"+m_jdata["managedWorker"] and jdata["roundId"]=="round"+m_jdata["roundId"]:
            return jdata

def read_survey(jDatas,FILEPATH):
    m_evas=[]
    ai_jDatas=read_json(FILEPATH+"/output.json")
    
    for jdata in jDatas:
        if jdata["surveyType"]=="WORKER_EVALUATION":
            ai_jdata=match_m_ai(jdata,ai_jDatas)
#             print(ai_jdata)
            if ai_jdata==None:
                print("NO MATCH: ", jdata)
                continue
            m_eva=[jdata["player"], jdata["roundId"], jdata["managedWorker"], jdata["answer"], \
                   eva_str2int[ai_jdata["evaluation"]], \
                   ai_jdata["score_probabilities"][0], ai_jdata["score_probabilities"][1], ai_jdata["score_probabilities"][2]]
            m_evas.append(m_eva)
    return m_evas
    
def write_csv(FILEPATH, OUTPATH=None):
    m_evas=[]
    g = os.walk(FILEPATH+"/survey")  
    for path,dir_list,file_list in g:
        for file_name in sorted(file_list):
            print('\n\n'+file_name)
            if file_name==".DS_Store":
                continue
            jsonPath = os.path.join(path, file_name)
            jDatas=read_json(jsonPath)
            m_evas=m_evas+read_survey(jDatas,FILEPATH)
            
    if OUTPATH==None:
        OUTPATH=FILEPATH
    path = OUTPATH+"/eva_compare.csv"
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["manager","round", "worker", "m_eva", "ai_eva", "p1", "p2", "p3"]
        csv_write.writerow(csv_head)
        for m_eva in m_evas:
            csv_write.writerow(m_eva)  
    f.close()


#add features 
FEATURES_NAME_0to5=["inventory_firstGainOrder", "inventory_firstGainStep",\
            "inventory_accum_reward", "inventory_rewardedGainStep", "inventory_rewardedGainOrder"]
FEATURES_NAME_5to20=["if_iron_axe", "if_stone_axe", "if_wooden_axe", "sparse_reward", "dense_reward", \
            "attack_effi", "attack_ratio", "attack_equipped", "camera_mov_ratio", "position_mov_ratio", \
            "torch_placed", "cobblestone_placed", "dirt_placed", "stone_placed", "if_smelt_coal"]

def read_survey_features(jDatas,FILEPATH):
    m_evas=[]
    ai_jDatas=read_json(FILEPATH+"/features.json")
    
    for jdata in jDatas:
        if jdata["surveyType"]=="WORKER_EVALUATION":
            ai_jdata=match_m_ai(jdata,ai_jDatas)
#             print(ai_jdata)
            if ai_jdata==None:
                print("NO MATCH: ", jdata)
                continue
            m_eva=[jdata["player"], jdata["roundId"], jdata["managedWorker"], jdata["answer"], \
                   eva_str2int[ai_jdata["evaluation"]], \
                   ai_jdata["score_probabilities"][0], ai_jdata["score_probabilities"][1], ai_jdata["score_probabilities"][2]]
            
            #add features
            for i in range(5):
                m_eva.append(ai_jdata["X_features"][i])
            for fea in FEATURES_NAME_5to20:
                m_eva.append(ai_jdata[fea])
            for i in range(5):
                for j in range(18):
                    m_eva.append(ai_jdata[FEATURES_NAME_0to5[i]][j])
            m_evas.append(m_eva)
    return m_evas
    
def write_csv_features(FILEPATH, OUTPATH=None):
    m_evas=[]
    g = os.walk(FILEPATH+"/survey")  
    for path,dir_list,file_list in g:
        for file_name in sorted(file_list):
            print('\n\n'+file_name)
            if file_name==".DS_Store":
                continue
            jsonPath = os.path.join(path, file_name)
            jDatas=read_json(jsonPath)
            m_evas=m_evas+read_survey_features(jDatas,FILEPATH)
            
    if OUTPATH==None:
        OUTPATH=FILEPATH
    path = OUTPATH+"/fea_m_compare.csv"
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["manager","round", "worker", "m_eva", "ai_eva", "p1", "p2", "p3"]+FEATURES_NAME_0to5+FEATURES_NAME_5to20
        for i in range(5):
            for j in range(18):
                csv_head+=[FEATURES_NAME_0to5[i]+"_"+str(j)]
        csv_write.writerow(csv_head)
        for m_eva in m_evas:
            csv_write.writerow(m_eva)  
    f.close()


if __name__ == "__main__":
    FILEPATH = sys.argv[1]
    write_csv_features(FILEPATH)