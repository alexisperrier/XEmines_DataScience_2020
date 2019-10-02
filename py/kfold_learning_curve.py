'''
Demonstration of overfitting detection with learning curve and 3 fold cross validation
Find the optimal degree of the polynomial regression
'''
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

df = pd.read_csv('../data/ozone.csv')

# remove missing values
df = df.dropna( subset= ['Ozone', 'Wind']   )
# Shuffle
df = df.sample(frac = 1, random_state = 8).reset_index(drop = True)
# create new polynimial variables up to degree 4
df['Wind2'] = df.Wind**2
df['Wind3'] = df.Wind**3
df['Wind4'] = df.Wind**4



# Manually create 3 fold indexes. This can obviously be done more simply with scikit-learn
idx1 = df[0: int(116/3)].index
idx2 = df[int(116/3): int(2*116/3 ) ].index
idx3 = df[int(2*116/3 ): ].index
idx = [idx1, idx2, idx3]

# p = 1
def train_test_mse(model):
    print('-----');
    print('Model: {}'.format(model));
    mse_train = []
    mse_test = []
    for k in range(3):
        test  =  df[ df.index.isin(idx[k]) ]
        train =  df[ ~df.index.isin(idx[k]) ]
        res   = smf.ols(model, data = train   ).fit()

        score_train = mse( train.Ozone, res.fittedvalues )
        score_test  = mse( test.Ozone, res.predict(test)   )
        mse_train.append( score_train  )
        mse_test.append( score_test )
        print("\t Fold {} MSE Train {:.3f} MSE Ttest {:.3f}".format(k, score_train, score_test))

    print("avg MSE Train: {:.2f}  ".format( np.mean(mse_train)  ))
    print("avg MSE Test: {:.2f}  ".format( np.mean(mse_test)  ))
    return np.mean(mse_train), np.mean(mse_test)

results = []

score_train, score_test = train_test_mse('Ozone ~ Wind')
results.append( {'train': score_train, 'test': score_test} )
score_train, score_test = train_test_mse('Ozone ~ Wind + Wind2')
results.append( {'train': score_train, 'test': score_test} )
score_train, score_test = train_test_mse('Ozone ~ Wind + Wind2 + Wind3')
results.append( {'train': score_train, 'test': score_test} )
score_train, score_test = train_test_mse('Ozone ~ Wind + Wind2 + Wind3 + Wind4')
results.append( {'train': score_train, 'test': score_test} )

results = pd.DataFrame(results)

fig, ax = plt.subplots(1,1, figsize = (6,6))
plt.plot(range(1,5),results['train'], label = 'train')
plt.plot(range(1,5),results['test'], label = 'test')
plt.title("polynomial regression Ozone ~ Wind + ... + Wind^p")
plt.xlabel('p: degree of polynomial')
plt.ylabel('MSE')
plt.legend()
plt.grid(0.4)
plt.show()
