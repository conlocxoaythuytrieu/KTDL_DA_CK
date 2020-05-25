import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import Pipeline
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.naive_bayes import BernoulliNB
import category_encoders as ce 
from sklearn.neighbors import KNeighborsClassifier
from ast import literal_eval
import pickle
from . import var_env

def dataset_numerical_train():
    train = pd.read_csv('./csv/dataset.csv',low_memory=False)

    cat_features = ['province','district','maCv', 'FIELD_8', 'FIELD_9','FIELD_10','FIELD_12','FIELD_13','FIELD_17','FIELD_18',
                'FIELD_19','FIELD_20','FIELD_22','FIELD_23','FIELD_24','FIELD_25','FIELD_26','FIELD_27',
                'FIELD_28','FIELD_29','FIELD_30','FIELD_31','FIELD_35','FIELD_36','FIELD_37',
                'FIELD_38','FIELD_39','FIELD_40','FIELD_41','FIELD_42','FIELD_43','FIELD_44',
                'FIELD_47','FIELD_48','FIELD_49']
    
    cat_features_remove = cat_features + ['id']
    target_enc = ce.CatBoostEncoder(cols=cat_features)
    
    target_enc.fit(train[cat_features], train['label'])

    train = train.join(target_enc.transform(train[cat_features]).add_suffix('_process'))

    train = train.drop(columns = cat_features_remove)

    f7_array = train['FIELD_7'].apply(lambda x: '[]' if x != x else x).apply(literal_eval)
    train['FIELD_7'] = f7_array.apply(len)

    train = train.replace(to_replace = 'None', value = np.nan) 

    train.to_csv('./csv/preprocess/numerical_train.csv')

    return True

def detection_outlier(df,cols):
    
    outlier_indices = np.array([])
    
    for col in cols:
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        outlier_indices = np.append(outlier_indices,outlier_list_col.values)         

    return np.unique(outlier_indices)

def deadling_missing_value_train():
   
    train = pd.read_csv('./csv/preprocess/numerical_train.csv',index_col=0)
   
    my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    convert_tool = my_imputer.fit(train)

    train = pd.DataFrame(convert_tool.transform(train), columns = train.columns)
    
    train.to_csv('./csv/preprocess/missing_train.csv')
    return True

def feture_extraction():
    train = pd.read_csv('./csv/preprocess/missing_train.csv',index_col=0)
    
    train_label = pd.DataFrame({'ID':train['label']})
    train = train.drop(columns = ['label'])
    
   
    special_column = "FIELD_55".split(" ")
    index_outlier_data = detection_outlier(train,special_column).astype(int)
    
    train = train.drop(index_outlier_data, axis = 0).reset_index(drop=True)
    train_label = train_label.drop(index_outlier_data, axis = 0).reset_index(drop=True)


    scaler = StandardScaler()
    scaler.fit(train)
    train = pd.DataFrame(scaler.transform(train),columns = train.columns)

    train_label.to_csv('./csv/preprocess/train_label.csv')
    train.to_csv('./csv/preprocess/feature_extraction.csv')

    return True
"""
    training full option
"""
def preprocessing(train,test):
    cat_features = ['province','district','maCv', 'FIELD_8', 'FIELD_9','FIELD_10','FIELD_12','FIELD_13','FIELD_17','FIELD_18',
                'FIELD_19','FIELD_20','FIELD_22','FIELD_23','FIELD_24','FIELD_25','FIELD_26','FIELD_27',
                'FIELD_28','FIELD_29','FIELD_30','FIELD_31','FIELD_35','FIELD_36','FIELD_37',
                'FIELD_38','FIELD_39','FIELD_40','FIELD_41','FIELD_42','FIELD_43','FIELD_44',
                'FIELD_47','FIELD_48','FIELD_49']
    # End catBoostEncoder We must remove columns and label from training data and test data 
    
    cat_features_remove = cat_features 
    target_enc = ce.CatBoostEncoder(cols=cat_features)
    
    target_enc.fit(train[cat_features], train['label'])

    train = train.join(target_enc.transform(train[cat_features]).add_suffix('_process'))
    test = test.join(target_enc.transform(test[cat_features]).add_suffix('_process'))

    train = train.drop(columns = cat_features_remove)
    test = test.drop(columns = cat_features_remove)
    
    train = train.replace(to_replace = 'None', value = np.nan) 
    test = test.replace(to_replace = 'None', value = np.nan)
    
    my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    convert_tool = my_imputer.fit(train)

    train = pd.DataFrame(convert_tool.transform(train), columns = train.columns)
    test  = pd.DataFrame(convert_tool.transform(test),  columns = test.columns)
    
    special_column = "FIELD_55".split(" ")
    index_outlier_data = detection_outlier(train,special_column).astype(int)
    
    
    train = train.drop(index_outlier_data, axis = 0).reset_index(drop=True)
    
    train_label = train['label']
   # print(train[index_outlier_data.astype(int)])
    train = train.drop(columns = ['label'])
    test  = test.drop(columns = ['label'])
    print(len(index_outlier_data)/len(train))
    scaler = StandardScaler()
    scaler.fit(train)
    train = pd.DataFrame(scaler.transform(train),columns = train.columns)
    test = pd.DataFrame(scaler.transform(test),columns = test.columns)

    return train,train_label,test
def calculate_f1_score(measure):
    p = measure[0][0]/(measure[0][0]+measure[0][1])
    r = measure[0][0]/(measure[0][0]+measure[1][0])
    return p,r,2*p*r/(p+r)
def transform_average_result(pres):
    averager_pre = []
    for value in pres:
        if value>.5:
            averager_pre.append(1)
        else:
            averager_pre.append(0)
    return averager_pre
def training():
    data = pd.read_csv('./csv/dataset.csv',index_col = 'id',low_memory=False)
    original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(data, data['label'], test_size=0.2, random_state=42)
    num_round = 10
    kfold = StratifiedKFold(n_splits=num_round, random_state=None, shuffle=False)
    id =0
    split_data ={}
    for train_id,test_id in kfold.split(original_Xtrain,original_ytrain):
        split_data['train'+str(id)] = train_id
        split_data['valid'+str(id)] = test_id
        id+=1
    
    f7_array = original_Xtrain['FIELD_7'].apply(lambda x: '[]' if x != x else x).apply(literal_eval)
    original_Xtrain['FIELD_7'] = f7_array.apply(len)

    f7_array = original_Xtest['FIELD_7'].apply(lambda x: '[]' if x != x else x).apply(literal_eval)
    original_Xtest['FIELD_7'] = f7_array.apply(len)

    original_Xtrain_,original_Xtrain_label_,original_Xtest = preprocessing(original_Xtrain,original_Xtest)

    num_round = 10
    process_data= {}

    for id in range(num_round):
        process_data['train'+str(id)],process_data['train_label'+str(id)],process_data['valid'+str(id)] = preprocessing(original_Xtrain.iloc[split_data['train'+str(id)]],original_Xtrain.iloc[split_data['valid'+str(id)]])

    result_valid  = {} # save result
    result_orginal = {}
    clssifers ={
        "KNearest": KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform'),
        "LogisticRegression" : LogisticRegression(max_iter = 1000000,penalty = 'l2'),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }
    
    for key,clf in clssifers.items():
        average_original_test = 0.0
        average_valid_test = 0.0
        average_private_test = 0.0
        print(key)
        for id in range(0,num_round):
            
            idx = split_data['valid'+str(id)]
            pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'),clf)
            model = pipeline.fit(process_data['train'+str(id)],process_data['train_label'+str(id)])
            
            score_valid_test = model.predict(process_data['valid'+str(id)])
            score_original_test = model.predict(original_Xtest)
            
            average_valid_test     += score_valid_test/num_round
            average_original_test  += score_original_test/num_round
        
        
        valid  = transform_average_result(average_valid_test)
        orginal = transform_average_result(average_original_test)
    
        result_valid[key] = metrics.confusion_matrix(valid,original_ytrain.iloc[idx])
        result_orginal[key] = metrics.confusion_matrix(orginal,original_ytest)
    pickle.dump(clssifers['KNearest'], open('./csv/model/KNearest.pkl', 'wb'))
    pickle.dump(clssifers['LogisticRegression'], open('./csv/model/LogisticRegression.pkl', 'wb'))
    pickle.dump(clssifers['DecisionTreeClassifier'], open('./csv/model/DecisionTreeClassifier.pkl', 'wb'))
    return True
    