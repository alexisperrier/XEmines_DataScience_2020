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
    df.head()
    df.shape

    # selectionner les colonnes numeriques : float et int
    num_variables = [col for col in df.columns
        if df[col].dtypes in  ['int64', 'float64'] ]
    # ne garder que les variables sans valeurs manquantes et pour les int qui ont beaucoup de valeurs differentes
    # for v in num_variables:
        # print( " var: {} \t NaN :{} scope {} ".format(v ,
        # df[df[v].isna()].shape[0],
        # df[v].value_counts().shape[0]   )   )

    features = [ v for v in num_variables
            if (df[df[v].isna()].shape[0] < 10) # missing values < 10
            &  (df[v].value_counts().shape[0] > 10) # more than 10 different values
            & (v not in ['Id']  ) ]

    df = df[features].dropna()
    df = df.sample(frac = 1, random_state = 88).reset_index(drop = True)
    print("df.shape: {} \nfeatures {}".format(df.shape, features))
    X = df.drop(columns = 'SalePrice').values
    y = df.SalePrice

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
    valid_features = features.copy()
    valid_features.remove('SalePrice')
    mdl.fit(X,y)

    vdf = pd.read_csv('./../data/ames_test.csv')
    vdf = vdf[valid_features]
    # replace missing values by average of variable
    for v in valid_features:
        if vdf[vdf[v].isna()].shape[0] > 0:
            # print("v {} mean {}".format(v, np.mean( vdf[v] )))
            vdf.loc[vdf[v].isna(), v] = np.mean( vdf[v] )

    X_valid = vdf.values
    y_valid = mdl.predict(X_valid)

    kaggle_submit(vdf,y_valid, '../kaggle/bagging_tree_04.csv')
