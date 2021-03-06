'''
Demo of stochastic gradient descent on Ames Housing dataset
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def log_rmse(yhat, ytrue):
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
    for v in num_variables:
        print( " var: {} \t NaN :{} scope {} ".format(v ,
        df[df[v].isna()].shape[0],
        df[v].value_counts().shape[0]   )   )

    features = [ v for v in num_variables
            if (df[df[v].isna()].shape[0] < 100) # missing values < 10
            &  (df[v].value_counts().shape[0] > 5) # more than 10 different values
            & (v not in ['Id']  ) ]

    df = df[features].dropna()
    df = df.sample(frac = 1).reset_index(drop = True)
    print("df.shape: {} \nfeatures {}".format(df.shape, features))
    X = df.drop(columns = 'SalePrice').values
    y = df.SalePrice
    avg_price = np.mean(y)

    # Baseline, simple regression
    mdl = LinearRegression()

    # K fold cross validation
    scores = []
    kf = KFold(n_splits=k_fold)
    k = 1
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test  = X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mdl.fit(X_train,y_train)
        # yhat_test = mdl.predict(X_test)
        yhat_test = []
        for s in mdl.predict(X_test):
            if s < 0:
                yhat_test.append(avg_price)
            else:
                yhat_test.append(s)
                
        score = log_rmse(  yhat_test,  y_test )
        print("\t score {:.3f}".format(score))
        scores.append(score)

    print("avg score {:.3f} std {:.3f}".format(np.mean(scores), np.std(scores)  ))

    '''
    Prediction sur le test
    '''

    valid_features = features.copy()
    valid_features.remove('SalePrice')
    # retrain mdl on all samples
    mdl.fit(X,y)


    vdf = pd.read_csv('./../data/ames_test.csv')
    vdf = vdf[valid_features]
    # replace missing values by average of variable
    for v in valid_features:
        if vdf[vdf[v].isna()].shape[0] > 0:
            print("v {} mean {}".format(v, np.mean( vdf[v] )))
            vdf.loc[vdf[v].isna(), v] = np.mean( vdf[v] )

    X_valid = vdf.values
    y_valid = []
    for s in mdl.predict(X_valid):
        if s < 0:
            y_valid.append(avg_price)
        else:
            y_valid.append(s)


    kaggle_submit(vdf,y_valid, '../kaggle/lin_reg_03.csv')
