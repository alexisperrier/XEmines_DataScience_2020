'''
Demo of bagging with regression tree on Ames Housing dataset
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

def log_rmse(yhat, ytrue):
    yhat = [ np.max([x,1]) for x in yhat  ]
    return np.sqrt( mean_squared_error ( np.log(yhat), np.log(ytrue) ))

def kaggle_submit(vdf,y_valid, filename):
    results = pd.DataFrame(columns = ['Id', 'SalePrice'])
    # Kaggle veut que la colonne index commence par 1461
    results['Id'] = vdf.index + 1461

    results['SalePrice'] = y_valid

    # écrire le resultats dans le fichier csv
    results.to_csv(filename, index = False)


if __name__ == "__main__":
    k_fold = 5

    df = pd.read_csv('./../data/ames_train.csv')
    vdf = pd.read_csv('./../data/ames_test.csv')

    df.head()
    df.shape

    # selectionner les colonnes numeriques : float et int
    num_features = [col for col in df.columns
        if df[col].dtypes in  ['int64', 'float64'] ]

    num_features = [ v for v in num_features if (v not in ['Id','SalePrice']  ) ]

    for v in num_features:
        val = np.mean( df[v] + vdf[v] )
        df[v].fillna( val , inplace = True)
        vdf[v].fillna( val , inplace = True )


    # categorical_features
    str_features = [ c for c in df.columns if df[c].dtypes == 'O'  ]
    # replace missing values with : unknown
    for c in str_features:
        df[c]   = df[c].fillna('unknown')
        vdf[c]  = vdf[c].fillna('unknown')
        le = LabelEncoder()
        le.fit(pd.concat([df[c], vdf[c]]))
        df[c]   = le.transform(df[c])
        vdf[c]  = le.transform(vdf[c])

    features = num_features + str_features

    df = df[features +  ['SalePrice']].dropna()

    df = df.sample(frac = 1, random_state = 88).reset_index(drop = True)
    vdf = vdf[features]

    print("df.shape: {} \nfeatures {}".format(df.shape, features))
    X = df.drop(columns = 'SalePrice').values
    y = df.SalePrice

    # Baseline, simple regression

    # base_mdl = LinearRegression()
    base_mdl = DecisionTreeRegressor(max_depth= 10)
    # mdl = base_mdl
    mdl = BaggingRegressor(base_mdl,
            n_estimators = 200,
            max_samples  = 1.0,
            max_features = 1.0,
            bootstrap    = True,
            bootstrap_features = True,
            oob_score = True,
            random_state = 88)

    # K fold cross validation
    scores, scores_train = [], []
    kf = KFold(n_splits = k_fold)

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test  = X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mdl.fit(X_train,y_train)
        yhat_test =  mdl.predict(X_test)
        yhat_train =  mdl.predict(X_train)
        score = log_rmse(  yhat_test,  y_test )
        score_train = log_rmse(  yhat_train,  y_train )
        print("\t score test {:.3f} train {:.3f}  ".format(score, score_train))
        scores.append(score)
        scores_train.append(score_train)

    print("avg score {:.3f} std {:.3f} train {:.3f}".format(np.mean(scores), np.std(scores), np.mean(scores_train)  ))

    # retrain mdl on all samples
    mdl.fit(X,y)

    # replace missing values by average of variable
    X_valid = vdf.values
    y_valid = mdl.predict(X_valid)

    kaggle_submit(vdf,y_valid, '../kaggle/bagging_tree_05.csv')
