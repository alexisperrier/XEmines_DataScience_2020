import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

pd.options.display.max_columns = None

if __name__ == '__main__':

    '''
    load data
    '''

    df = pd.read_csv('../data/titanic.csv')

    df.drop(columns = ['ticket', 'boat', 'body','cabin','name', 'home.dest'], inplace = True )
    # df.drop(columns = ['fare','ticket', 'boat', 'body','cabin','name', 'home.dest'], inplace = True )

    # print(df.head())
    # print(df.shape)
    # print(df.describe())

    '''
    Missing values
    '''
    df['embarked'].fillna( 'S' , inplace = True)
    df['age'].fillna( np.mean(df.age) , inplace = True)
    df['fare'].fillna( np.mean(df.fare) , inplace = True)

    '''
    Variables catégoriques
    '''
    le = LabelEncoder()
    df['embarked'] = le.fit_transform(df.embarked)
    df['sex'] = le.fit_transform(df.sex)

    # print(df.head())

    ''' X ,y '''
    X = df.drop(columns = 'survived').values
    y = df.survived

    '''
    Model
    '''
    # mdl = DecisionTreeClassifier(max_depth= 10)
    max_depths = [2,3,4,5,6,7,10,15,20]
    n_estimators = [30,40,50,75,100,150,200,300]
    k_fold = 4


    res = []
    for md in max_depths:
        # print()
        for nest in n_estimators:

            mdl = RandomForestClassifier(max_depth= md,
                n_estimators = nest,
                oob_score = True,
                max_features='auto',
                random_state=88)
            '''
            Kfold cross validation
            '''
            scores_test, scores_train = [], []
            kf = KFold(n_splits = k_fold)

            for train_index, test_index in kf.split(X):
                X_train = X[train_index]
                X_test  = X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                mdl.fit(X_train,y_train)
                score_test = mdl.score(  X_test,  y_test )
                score_train = mdl.score(  X_train,  y_train )
                # print("\t score test {:.3f} train {:.3f}  ".format(score_test, score_train))
                scores_test.append(score_test)
                scores_train.append(score_train)

            # print("md {}, nest {},  test {:.3f} +- {:.2f} train {:.3f}".format(md,nest,np.mean(scores_test), np.std(scores_test), np.mean(scores_train)  ))
            res.append({
                'depth': md,
                'nest': nest,
                'train': np.mean(scores_train),
                'test': np.mean(scores_test),
            })

    res = pd.DataFrame(res)


    if False:
        fig,ax = plt.subplots(1,1, figsize = (9,6))
        plt.plot(res.depth.unique(), res.groupby(by = 'depth').mean().reset_index().train, label = 'train')
        plt.plot(res.depth.unique(), res.groupby(by = 'depth').mean().reset_index().test, label = 'test')
        plt.xlabel('depth')
        plt.ylabel('accuracy')
        plt.title("train, test accuracy vs depth")
        plt.ylim(0.5,1)
        plt.grid(alpha = 0.3)
        plt.legend()
        plt.show()

        fig,ax = plt.subplots(1,1, figsize = (9,6))
        plt.plot(res.nest.unique(), res.groupby(by = 'nest').mean().reset_index().train, label = 'train')
        plt.plot(res.nest.unique(), res.groupby(by = 'nest').mean().reset_index().test, label = 'test')
        plt.xlabel('nest')
        plt.ylabel('accuracy')
        plt.title("train, test accuracy vs n estimators")
        plt.ylim(0.5,1)
        plt.grid(alpha = 0.3)
        plt.legend()
        plt.show()

        cond = res.depth == 10
        fig,ax = plt.subplots(1,1, figsize = (9,6))
        plt.plot(res[cond].nest.unique(), res[cond].train, label = 'train')
        plt.plot(res[cond].nest.unique(), res[cond].test, label = 'test')
        plt.title("train, test accuracy pour depth = 10")
        plt.xlabel('n estimator')
        plt.ylabel('accuracy')
        plt.ylim(0.5,1)
        plt.grid(alpha = 0.3)
        plt.legend()
        plt.show()



# ---------------
