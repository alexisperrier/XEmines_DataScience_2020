'''
Demonstration of overfitting detection with learning curve and 3 fold cross validation
Find the optimal degree of the polynomial regression
'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

df = pd.read_csv('../data/ozone.csv')

# remove missing values
df = df.dropna( subset= ['Ozone', 'Wind','Temp']   )
df['Wind2'] = df.Wind**2
df['Wind3'] = df.Wind**3
df['Wind4'] = df.Wind**4

df['Temp2'] = df.Temp**2
df['Temp3'] = df.Temp**3
df['Temp4'] = df.Temp**4

# Shuffle
df = df.sample(frac = 1, random_state = 8).reset_index(drop = True)

# Manually create 3 fold indexes.
idx1 = df[0: int(116/3)].index
idx2 = df[int(116/3): int(2*116/3 ) ].index
idx3 = df[int(2*116/3 ): ].index
idx  = [idx1, idx2, idx3]


def cv_mse(features,alpha):
    mse_train = []
    mse_test  = []

    # cross validation avec  scikit learn
    k_fold = 2

    X = df[features].values
    y = df.Ozone

    kf = KFold(n_splits=k_fold)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mdl = Ridge(alpha = alpha)
        mdl.fit(X_train,y_train)

        yhat_train = mdl.predict( X_train  )
        yhat_test = mdl.predict( X_test  )

        score_train = mse( y_train, yhat_train )
        score_test  = mse( y_test, yhat_test  )

        mse_train.append( score_train  )
        mse_test.append( score_test )
        # print("\t MSE Train {:.3f} MSE Ttest {:.3f}".format( score_train, score_test))

    print("[{:.2f}] avg MSE Train: {:.2f} MSE Test: {:.2f} std {:.2f} {}".format(alpha, np.mean(mse_train) ,  np.mean(mse_test),  np.std(mse_test), features  ))
    return np.mean(mse_train), np.mean(mse_test)

# comparing 2 models
for alpha in np.linspace(0,2,11):
    cv_mse(['Wind','Temp'], alpha)


for alpha in np.linspace(0,2,11):
    cv_mse(['Wind','Wind2','Temp','Temp2'], alpha)







# ----------------------------------
