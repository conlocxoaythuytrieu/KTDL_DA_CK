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

    train.to_csv('./preprocess/numerical_train.csv')

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
   
    train = pd.read_csv('./preprocess/numerical_train.csv',index_col=0)
   
    my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    convert_tool = my_imputer.fit(train)

    train = pd.DataFrame(convert_tool.transform(train), columns = train.columns)
   
    train.to_csv('./preprocess/missing_train.csv')
    return True

def feture_extraction():
    train = pd.read_csv('./preprocess/missing_train.csv',index_col=0)

    train_label = pd.DataFrame({'ID':train['label']})
    
    train = train.drop(columns = ['label'])

    special_column = "FIELD_55".split(" ")
    index_outlier_data = detection_outlier(train,special_column).astype(int)
    
    train = train.drop(index_outlier_data, axis = 0).reset_index(drop=True)
    train_label = train_label.drop(index_outlier_data, axis = 0).reset_index(drop=True)

    train_label.to_csv('./preprocess/feature_extraction.csv')
    train.to_csv('./preprocess/feature_extraction.csv')

    return True

