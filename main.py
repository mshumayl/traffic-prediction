#%%

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import RepeatedKFold, GridSearchCV
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

from sklearn.preprocessing import OneHotEncoder

import time

#%%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

#%%
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

#%%
def create_time_features(df):
    """
    Takes in a DataFrame with `time` column and creates `hour`, `day`, `day_of_year`, and checks if `is_weekend` 
    """
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day_name()
    df['day_of_year'] = df['time'].dt.dayofyear


    weekend_cond = [
        df['day'] == 'Saturday',
        df['day'] == 'Sunday',
    ]

    outputs = [1, 1]
    df['is_weekend'] = pd.Series(np.select(weekend_cond, outputs, 0)) #numpy operation is faster than lambda .apply() per row
    
    return df

#%%
def apply_ohe(df, column: str):
    """
    Applies One-Hot Encoding on the specified column
    """
    ohe = OneHotEncoder()
    enc = ohe.fit_transform(df[column].values.reshape(-1,1))
    enc_df = pd.DataFrame(enc.toarray())
    
    return enc_df
    
def append_ohe_columns(df, enc_df, column: str):
    """
    Renames One-Hot Encoded Columns and append to the main DataFrame
    """
    dir_list = df[column].unique().tolist()
    df = df.join(enc_df)
    
    return df

#%%

train = create_time_features(train)

enc_direction = apply_ohe(train, 'direction')
train = append_ohe_columns(train, enc_direction, 'direction')

enc_day = apply_ohe(train, 'day')
train = append_ohe_columns(train, enc_day, 'day')

train = train.set_index('time')
train.drop(['direction', 'day', 'row_id'], axis=1, inplace=True)
#%%
TARGET_VARS = ['congestion']
SEED = 2022
N_FOLDS = 3
N_REPEATS = 1

#%%
#Feature-target split
train_Y = pd.concat([train.pop(target) for target in TARGET_VARS], axis=1)
train_X = train

#%%

#Define model hyperparameters and CV parameters
hyperparams_gb = {'estimator__learning_rate': [0.05, 0.1],
                   'estimator__max_depth': [3, 5],
                   'estimator__subsample': [0.5, 0.75],
                   'estimator__n_estimators': [200]}

hyperparams_rf = {'estimator__bootstrap': [True],
                  'estimator__max_depth': [5, 10, None],
                  'estimator__max_features': ['auto', 'log2'],
                  'estimator__n_estimators': [5, 7, 9]}

hyperparams_lr = {'estimator__fit_intercept':[True, False]}


cv_params = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS)

#Initialize VotingRegressor components
reg1 = GradientBoostingRegressor(random_state=SEED)
reg2 = RandomForestRegressor(random_state=SEED)
reg3 = LinearRegression()

#Initialize CV for GB
cv_model_gb = MultiOutputRegressor(reg1)
crossval_gb = GridSearchCV(cv_model_gb, hyperparams_gb, scoring='neg_mean_squared_error', cv=cv_params, n_jobs=-1)

with open("pickle_gb_gs.pkl", 'wb') as file:
    pickle.dump(crossval_gb, file)

crossval_gb.fit(train_X, train_Y)

with open("pickle_gb_fit.pkl", 'wb') as file:
    pickle.dump(crossval_gb, file)

#Initialize CV for RF
cv_model_rf = MultiOutputRegressor(reg2)
crossval_rf = GridSearchCV(cv_model_rf, hyperparams_rf, scoring='neg_mean_squared_error', cv=cv_params, n_jobs=-1)

with open("pickle_rf_gs.pkl", 'wb') as file:
    pickle.dump(crossval_rf, file)

crossval_rf.fit(train_X, train_Y)

with open("pickle_rf_fit.pkl", 'wb') as file:
    pickle.dump(crossval_rf, file)

#Initialize CV for LR
cv_model_lr = MultiOutputRegressor(reg3)
crossval_lr = GridSearchCV(cv_model_lr, hyperparams_lr, scoring='neg_mean_squared_error', cv=cv_params, n_jobs=-1)

with open("pickle_lr_gs.pkl", 'wb') as file:
    pickle.dump(crossval_lr, file)
    
crossval_lr.fit(train_X, train_Y)

with open("pickle_lr_fit.pkl", 'wb') as file:
    pickle.dump(crossval_lr, file)

#Visualize CV error for Gradient Boosting
error = np.vstack([crossval_gb.cv_results_['split{}_test_score'.format(str(i))] for i in range(N_FOLDS*N_REPEATS)])
plt.figure(figsize=(16, 4))
plt.boxplot(error); plt.ylabel('neg_MSE')
plt.title('CV error for Gradient Boosting')

#Visualize CV error for Random Forests
error = np.vstack([crossval_rf.cv_results_['split{}_test_score'.format(str(i))] for i in range(N_FOLDS*N_REPEATS)])
plt.figure(figsize=(16, 4))
plt.boxplot(error); plt.ylabel('neg_MSE')
plt.title('CV error for Random Forests')

#Visualize CV error for Linear Regression
error = np.vstack([crossval_lr.cv_results_['split{}_test_score'.format(str(i))] for i in range(N_FOLDS*N_REPEATS)])
plt.figure(figsize=(16, 4))
plt.boxplot(error); plt.ylabel('neg_MSE')
plt.title('CV error for Linear Regression')

#%%

#Optimal params
opt_params_gb = crossval_gb.best_params_
opt_params_rf = crossval_rf.best_params_
opt_params_lr = crossval_lr.best_params_

#models
model_gb = GradientBoostingRegressor(learning_rate=opt_params_gb['estimator__learning_rate'],
                                     max_depth=opt_params_gb['estimator__max_depth'],
                                     subsample=opt_params_gb['estimator__subsample'],
                                     n_estimators=opt_params_gb['estimator__n_estimators'])

with open("pickle_gb_final.pkl", 'wb') as file:
    pickle.dump(model_gb, file)

model_rf = RandomForestRegressor(bootstrap=opt_params_rf['estimator__max_depth'],
                                 max_depth=opt_params_rf['estimator__max_depth'],
                                 max_features=opt_params_rf['estimator__max_features'],
                                 n_estimators=opt_params_rf['estimator__n_estimators'])

with open("pickle_rf_final.pkl", 'wb') as file:
    pickle.dump(model_rf, file)

model_lr = LinearRegression(fit_intercept=opt_params_lr['estimator__fit_intercept'])

with open("pickle_lr_final.pkl", 'wb') as file:
    pickle.dump(model_lr, file)

model_vr = MultiOutputRegressor(VotingRegressor(estimators=[('gb', model_gb), ('rf', model_rf), ('lr', model_lr)]))
model_vr.fit(train_X, train_Y)

with open("pickle_vr_final.pkl", 'wb') as file:
    pickle.dump(model_vr, file)

print('The optimal hyperparams for GradientBoostingRegressor are:\n', opt_params_gb)
print('The optimal hyperparams for RandomForestRegressor are:\n', opt_params_rf)
print('The optimal hyperparams for LinearRegression are:\n', opt_params_lr)

#%%

test = create_time_features(test)

enc_direction = apply_ohe(test, 'direction')
test = append_ohe_columns(test, enc_direction, 'direction')

enc_day = apply_ohe(test, 'day')
test = append_ohe_columns(test, enc_day, 'day')

test['Tuesday'] = 0
test['Wednesday'] = 0
test['Thursday'] = 0
test['Friday'] = 0
test['Saturday'] = 0
test['Sunday'] = 0

test = test.set_index('time')
test.drop(['direction', 'day', 'row_id'], axis=1, inplace=True)

#%%
#Load trained model
with open("../input/results/pickle_vr_final.pkl", "rb") as file:
    vr_model = pickle.load(file)

preds = vr_model.predict(test)
df = pd.DataFrame(preds)

sample_submission['congestion'] = df[0]
sample_submission.to_csv('submission.csv', index=False)