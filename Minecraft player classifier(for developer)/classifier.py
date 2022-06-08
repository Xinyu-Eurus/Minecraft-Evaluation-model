from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_text
import numpy as np
import os.path
from pathlib import Path

FEATURES_NUM=20
reward_mask=[0,16,4,0,32,0,128,64,256,1,2,4,0,0,32,0,0,8]
#default set of input_tuple
input_tuple=[0]*FEATURES_NUM
tuple0to4=[0]*18
input_tuple[0]=input_tuple[1]=input_tuple[2]=input_tuple[3]=input_tuple[4]=tuple0to4
input_tuple[5]=input_tuple[6]=input_tuple[7]=input_tuple[19]=True

def get_model():
    clf = load('./models/RF-optim.model')
    # clf = load('./models/RF-origin.model')
    return clf

def print_decision_tree(clf):
    estimator=clf.estimators_[0]
    text_representation = export_text(estimator)
    print(text_representation)

    # uncomment the following lines to show the visualized tree
    # plt.figure(figsize=(50,50))
    # plot_tree(estimator, filled=True)
    # plt.savefig('./rf-0.jpeg')
    # plt.show()
    return text_representation

def predict(clf, X_features):
    prediction = clf.predict(X_features)
    return prediction


def get_X_features(input_tuple):
    X_features = np.zeros((1, FEATURES_NUM))
    #get labels of 5 k-means
    km=[None]*5
    for i in range(5):
        km[i] = load('./models/km_'+str(i)+'.model')

    seq1=np.zeros((1,18))
    seq0=np.zeros((1,18))
    seq2=np.zeros((1,18))
    for i in range(18):
        seq1[0,i]=input_tuple[1][i] if reward_mask[i]!=0 else 0
        seq0[0,i]=input_tuple[0][i] if reward_mask[i]!=0 else 0
        seq2[0,i]=input_tuple[2][i]*reward_mask[i]

    X_features[:,0]=km[0].predict(input_tuple[0].reshape(-1,18)) 
    X_features[:,1]=km[1].predict(input_tuple[1].reshape(-1,18)) 
    X_features[:,2]=km[2].predict(seq2) 
    X_features[:,3]=km[3].predict(seq1) 
    X_features[:,4]=km[4].predict(seq0) 

    X_features[:,5]=1 if ((input_tuple[5]==bool and input_tuple[5]) or input_tuple[5]==1) else 0
    X_features[:,6]=1 if ((input_tuple[6]==bool and input_tuple[6]) or input_tuple[6]==1) else 0
    X_features[:,7]=1 if ((input_tuple[7]==bool and input_tuple[7]) or input_tuple[7]==1) else 0
    X_features[:,8]=input_tuple[8]
    X_features[:,9]=input_tuple[9]
    X_features[:,10]=input_tuple[10]
    X_features[:,11]=input_tuple[11]
    X_features[:,12]=input_tuple[12]
    X_features[:,13]=input_tuple[13]
    X_features[:,14]=input_tuple[14]
    X_features[:,15]=input_tuple[15]
    X_features[:,16]=input_tuple[16]
    X_features[:,17]=input_tuple[17]
    X_features[:,18]=input_tuple[18]
    X_features[:,19]=1 if ((input_tuple[19]==bool and input_tuple[19]) or input_tuple[19]==1) else 0

    return X_features

def check_input_tuple(input_tuple):
    assert(len(input_tuple)==20)
    assert(len(input_tuple[0])==18 and len(input_tuple[1])==18 and len(input_tuple[2])==18)
    
    
#main
if __name__ == "__main__":
    clf=get_model()
    print("model loaded successfully!")
    
    print("\n###########################")
    print("# 1: print_decision_tree  #")
    print("# 2: check_input_tuple    #")
    print("# 3: demo_predict         #")
    print("# 0: exit                 #")
    print("###########################")
    op_number = input("please input operation number:")
    
    while op_number!='0':
        if op_number=='1':
            print_decision_tree(clf)
        else:
            input_filepath=input("please input file path of the input_tuple:")
            if Path(input_filepath).is_file():
                input_tuple=np.load(input_filepath,allow_pickle=True)
            else:
                input_tuple=np.load("demo_data.npy",allow_pickle=True)

        if op_number=='2':        
            check_input_tuple(input_tuple)
        
        if op_number=='3':  
            X_features=get_X_features(input_tuple)
            predictions=predict(clf, X_features)
            print("predictions:",predictions)
        
        op_number = input("please input operation number:")

    print("Bye!")