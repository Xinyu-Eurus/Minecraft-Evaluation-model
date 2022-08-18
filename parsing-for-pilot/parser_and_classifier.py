from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_text
import numpy as np
import json
from pathlib import Path

FEATURES_NUM=20
#default set of input_tuple
input_tuple=[0]*FEATURES_NUM
tuple0to4=[0]*18
input_tuple[0]=input_tuple[1]=input_tuple[2]=input_tuple[3]=input_tuple[4]=tuple0to4
input_tuple[5]=input_tuple[6]=input_tuple[7]=input_tuple[19]=True

item2num={"COAL":0, "COBBLESTONE":1, "CRAFTING_TABLE":2, "DIRT":3, "FURNACE":4, "IRON_AXE":5, "IRON_INGOT":6,\
         "IRON_ORE":7, "IRON_PICKAXE":8, \
          "JUNGLE_LOG":9,"OAK_LOG":9,"DARK_OAK_LOG":9,"SPRUCE_LOG":9,"ACACIA_LOG":9,"BIRCH_LOG":9,"MANGROVE_LOG":9,\
         "JUNGLE_PLANKS":10,"OAK_PLANKS":10,"DARK_OAK_PLANKS":10,"SPRUCE_PLANKS":10,"ACACIA_PLANKS":10,"BIRCH_PLANKS":10,\
         "STICK":11, "STONE":12, \
         "STONE_AXE":13, "STONE_PICKAXE":14, "TORCH":15, "WOODEN_AXE":16, "WOODEN_PICKAXE":17} #for inventoryTrace #currently no "STONE"

reward_mask=[0,16,4,0,32,0,128,64,256,1,2,4,0,0,32,0,0,8]
reward_mask=np.array(reward_mask)
FILENAME0="./trace_Worker_768451.json"

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


def extractFeatures(jDatas):    
    inventoryDiffs=[]
    inventoryLists=[]
    steps=len(jDatas)
    inventory_firstGainOrder=np.zeros((18,)) #0
    order_count=1
    inventory_firstGainStep=np.zeros((18,)) #1
    inventory_accum=np.zeros((18,)) #2.0 (return: inventory_accum_reward)
    inventory_rewardedGainStep=np.zeros((18,)) #3
    inventory_rewardedGainOrder=np.zeros((18,)) #4
    
    attack_count=0
    equip_tool_atk_count=0
    
    cmr_mov_count=0
    pos_mov_count=0
    
    #inventory0
    inventoryTrace0=jDatas[0]["inventoryTrace"]
    inventorylist0=np.zeros((18,))
    for item in inventoryTrace0.keys():
        inventorylist0[item2num[item]] = inventoryTrace0[item]
    inventoryLists.append(inventorylist0)
    
    #movings0
    directionDifference0=jDatas[0]["directionDifference"]
    playerPosition0=jDatas[0]["playerPosition"]
    
    
    for i in range(1,len(jDatas)):
        jdata=jDatas[i]
        
        #inventory
        inventoryTrace=jdata["inventoryTrace"]
#         inventorydiff={inventorydiff[key]=inventoryTrace[key]-inventoryTrace0[key] for key in inventoryTrace.keys()}
        inventorylist=np.zeros((18,))
        for item in inventoryTrace.keys():
            inventorylist[item2num[item]] = inventoryTrace[item]
        inventoryLists.append(inventorylist)
        
        inventorydiff=inventorylist-inventorylist0
        inventoryDiffs.append(inventorydiff)
        inventorylist0=inventorylist
        
        if np.any(inventorydiff>0):
            inventory_accum+=np.where(inventorydiff>0,inventorydiff,0)
            nonz_idx=np.where(inventorydiff>0)
            for idx in nonz_idx: 
                if inventory_firstGainOrder[idx]==0:
                    inventory_firstGainOrder[idx]=order_count
                    order_count+=1
                    inventory_firstGainStep[idx]=i
        
        #attack
        isAttacking=jdata["isAttacking"]   
        if isAttacking:
            attack_count+=1
            recentlyEquippedGoalItem=jdata["recentlyEquippedGoalItem"]
            if recentlyEquippedGoalItem=="WOODEN_PICKAXE" or recentlyEquippedGoalItem=="STONE_PICKAXE":
                equip_tool_atk_count+=1
            
        #movings
        directionDifference=jdata["directionDifference"]
        playerPosition=jdata["playerPosition"]
        isJumping=jdata["isJumping"]
        isSneaking=jdata["isSneaking"]
        isSprinting=jdata["isSprinting"]
        
        if directionDifference["x"]-directionDifference0["x"]!=0 \
            or directionDifference["y"]-directionDifference0["y"]!=0 \
            or directionDifference["z"]-directionDifference0["z"]!=0:
            cmr_mov_count+=1
        if playerPosition["x"]-playerPosition0["x"]!=0 \
            or playerPosition["y"]-playerPosition0["y"]!=0 \
            or playerPosition["z"]-playerPosition0["z"]!=0 \
            or isJumping or isSneaking or isSprinting:
            pos_mov_count+=1
        directionDifference0=directionDifference
        playerPosition0=playerPosition
        
        #place
    
    #(inventory) seq 0-4
    inventory_accum_reward=[inventory_accum[x]*reward_mask[x] for x in range(18)]
    inventory_accum_reward=np.array(inventory_accum_reward)
    inventory_rewardedGainStep=np.where(reward_mask>0,inventory_firstGainStep,0)
    inventory_rewardedGainOrder=np.where(reward_mask>0,inventory_firstGainOrder,0)    
    
    #(inventory) if axes are made 5-7
    if_iron_axe = inventory_accum[5]>0
    if_stone_axe= inventory_accum[13]>0
    if_wooden_axe=inventory_accum[16]>0
    
    #(inventory) reward 8-9
    sparse_reward=np.sum(np.where(inventory_accum_reward>0,reward_mask,0))
    dense_reward=np.sum(inventory_accum_reward)
    
    #(attack) 10-12
    attack_effi=(inventory_accum[1]+inventory_accum[7]+inventory_accum[9])*1.0/attack_count #total_digged_items/attack
    attack_ratio=attack_count*1.0/len(jDatas) #attack/steps
    attack_equipped=equip_tool_atk_count*1.0/attack_count #attack_and_equipped_pickaxe/attack

    #(movings) 13-14
    camera_mov=cmr_mov_count*1.0/len(jDatas)
    position_mov=pos_mov_count*1.0/len(jDatas)
    
    #(place) 15-18
    torch_placed=inventory_accum[15]-(inventoryLists[steps-1][15]-inventoryLists[0][15])
    cobblestone_placed=inventory_accum[1]-(inventoryLists[steps-1][1]-inventoryLists[0][1])
    dirt_placed=inventory_accum[3]-(inventoryLists[steps-1][3]-inventoryLists[0][3])
    stone_placed=inventory_accum[12]-(inventoryLists[steps-1][12]-inventoryLists[0][12])
    if_smelt_coal=False
    
    print("inventory_firstGainOrder:",inventory_firstGainOrder)
    print("inventory_firstGainStep:",inventory_firstGainStep)
    print("inventory_accum:",inventory_accum)
    print("inventory_accum_reward:",inventory_accum_reward)
    print("inventory_rewardedGainStep:",inventory_rewardedGainStep)
    print("inventory_rewardedGainOrder:",inventory_rewardedGainOrder)
    
    print("if_iron_axe:",if_iron_axe)
    print("if_stone_axe:",if_stone_axe)
    print("if_wooden_axe:",if_wooden_axe)
    
    print("sparse_reward:",sparse_reward)
    print("dense_reward:",dense_reward)
    
    print("attack_effi:",attack_effi)
    print("attack_ratio:",attack_ratio)
    print("attack_equipped:",attack_equipped)

    print("camera_mov:",camera_mov)
    print("position_mov:",position_mov)
    
    print("place(torch, cobblestone, dirt, stone):", torch_placed, cobblestone_placed, dirt_placed, stone_placed)
    
    # a = np.concatenate((inventory_firstGainOrder, inventory_firstGainStep, inventory_accum, inventory_accum_reward,inventory_rewardedGainStep,inventory_rewardedGainOrder),axis=0)
    # np.savetxt("seqs.csv", a, delimiter=",")

    return (inventory_firstGainOrder, inventory_firstGainStep,\
            inventory_accum_reward, inventory_rewardedGainStep, inventory_rewardedGainOrder,\
            if_iron_axe, if_stone_axe, if_wooden_axe, sparse_reward, dense_reward, \
            attack_effi, attack_ratio, attack_equipped, camera_mov, position_mov, \
            torch_placed, cobblestone_placed, dirt_placed, stone_placed, if_smelt_coal)





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
        X_features[:,i]=km[i].predict(input_tuple[i].reshape(-1,18)) 
        print("X_features",i, X_features[:,i])
    
    X_features[:,5]=1 if ((input_tuple[5]==bool and input_tuple[5]) or input_tuple[5]==1) else 0
    X_features[:,6]=1 if ((input_tuple[6]==bool and input_tuple[6]) or input_tuple[6]==1) else 0
    X_features[:,7]=1 if ((input_tuple[7]==bool and input_tuple[7]) or input_tuple[7]==1) else 0

    for i in range(8,19):
        X_features[:,i]=input_tuple[i]

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
    # print("# 0: exit                 #")
    print("# 1: predict              #")
    print("# 2: demo_predict         #")
    print("# 3: print_decision_tree  #")
    print("###########################")
    op_number = input("please input operation number:")
    
    # while op_number!='0':
    if op_number=='3':
        print_decision_tree(clf)

    if op_number=='1': 
        FILENAME=input("please input file path of the input_tuple:")
        if Path(FILENAME).is_file():
            jDatas=read_json(FILENAME)
        else:
            jDatas=read_json(FILENAME0)     
            
        input_tuple=extractFeatures(jDatas)
        check_input_tuple(input_tuple)
        X_features=get_X_features(input_tuple)
        predictions=predict(clf, X_features)
        print("\npredictions:",predictions)

    if op_number=='2':
        input_tuple=np.load("demo_data.npy",allow_pickle=True)
        X_features=get_X_features(input_tuple)
        print("\npredictions:",predict(clf, X_features))
        
        # op_number = input("please input operation number:")

    print("Bye!")