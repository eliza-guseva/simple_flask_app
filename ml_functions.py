import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    average_precision_score,
)


categorical_cols = [
    'Gender', # binary, doesn't seem to matter
    'Customer Type', # binary, matters
    'Type of Travel', # binary, matters
    'Class', # binary, matters
] 
numerical_cols = [
    'Age', # matters
    'Flight Distance', # matter
    'Inflight wifi service', # matters
    'Departure/Arrival time convenient', # doesn't matter
    'Ease of Online booking', # matters
    'Gate location', # matters
    'Food and drink', # matters
    'Online boarding', # matters
    'Seat comfort', # matters
    'Inflight entertainment', # matters
    'On-board service', # matters
    'Leg room service', # matters
    'Baggage handling', # matters
    'Checkin service', # matters
    'Inflight service', # matters
    'Cleanliness', # matters
    'Arrival Delay in Minutes', # doesn't matter
    'Departure Delay in Minutes', # doesn't matter
]

# read and preprocess data

def read_csv(file_name):
    return pd.read_csv(file_name, index_col=0)

def preprocess_data(df):
    clean_df = df.drop(columns=['id'])
    return clean_df

# split data into train and test and fill missing values

def split_train_test(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(train_df.shape, test_df.shape)

    train_x = train_df.drop(columns=['satisfaction'])
    train_y = train_df['satisfaction']

    test_x = test_df.drop(columns=['satisfaction'])
    test_y = test_df['satisfaction']

    return train_x, train_y, test_x, test_y

def get_median(df, column_name):
    return df[column_name].fillna(df[column_name].median())

def fill_missing_values(df, median_dict):
    for column_name, median in median_dict.items():
        df[column_name] = df[column_name].fillna(median)
    return df


# feature engineering



def encode_y(y, is_train):
    if is_train:
        le = LabelEncoder()
        le.fit_transform(y)
        pickle.dump(le, open('le.pkl', 'wb'))
    else:
        le = pickle.load(open('le.pkl', 'rb'))
    return le.transform(y)

def one_hot_encode_categorical_cols(df, categorical_cols, ohe):
    categorical_train_x = ohe.transform(df[categorical_cols])
    categorical_cols_names = ohe.get_feature_names_out(categorical_cols)
    categorical_train_df = pd.DataFrame(
        categorical_train_x, 
        columns=categorical_cols_names,
        index=df.index,
        )
    original_df = df.drop(columns=categorical_cols)
    return pd.concat([original_df, categorical_train_df], axis=1)

def scale_data(df, scaler):
    scaler.transform(df)
    scaler.feature_names_in_
    return pd.DataFrame(
        scaler.transform(df), 
        columns=scaler.feature_names_in_, 
        index=df.index
        )
    
def feature_engineere_all(x_data, y, categorical_cols, is_train):
    if is_train:
        ohe = OneHotEncoder(sparse_output=False)
        ohe.fit(x_data[categorical_cols])
        pickle.dump(ohe, open('ohe.pkl', 'wb'))
        x_data = one_hot_encode_categorical_cols(x_data, categorical_cols, ohe)
        scaler = StandardScaler()
        scaler.fit(x_data)
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        x_data = scale_data(x_data, scaler)
        y = encode_y(y, is_train)
    else:
        ohe = pickle.load(open('ohe.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        x_data = one_hot_encode_categorical_cols(x_data, categorical_cols, ohe)
        x_data = scale_data(x_data, scaler)
        if y is not None:
            y = encode_y(y, is_train)           
    return x_data, y
        

def select_model(train_x, train_y, test_x, test_y):
    params = {
        'n_estimators': [50, 150],
        'max_depth': [None, 10, 20, 30]
    }
    
    x_combined = pd.concat([train_x, test_x])
    y_combined = np.concatenate([train_y, test_y])

    split_index = [-1] * len(train_x) + [0] * len(test_x)
    pds = PredefinedSplit(test_fold=split_index)

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=params,
        cv=pds,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(x_combined, y_combined)
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_} for accuracy {grid_search.best_score_}")
    return best_model, grid_search.best_params_, grid_search.best_score_

def train_model(model, train_x, train_y):
    model.fit(train_x, train_y)
    pickle.dump(model, open('best_model.pkl', 'wb'))
    return model


