import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

import shap

pd.options.display.max_columns = None

if __name__ == '__main__':

    '''
    load data
    '''

    df = pd.read_csv('../data/titanic.csv')

    df.drop(columns = ['ticket', 'boat', 'body','cabin','name', 'home.dest'], inplace = True )

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
    max_depth = 8
    n_estimators = 50
    mdl = RandomForestClassifier(max_depth= max_depth,
        n_estimators = n_estimators,
        oob_score = True,
        max_features='auto',
        random_state=88)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    mdl.fit(X_train,y_train)
    score_test = mdl.score(  X_test,  y_test )
    score_train = mdl.score(  X_train,  y_train )

    print("test {:.3f} +- {:.2f} train {:.3f}".format(np.mean(score_test), np.std(score_test), np.mean(score_train)  ))

    for c, i in zip( df.drop(columns = 'survived').columns, mdl.feature_importances_   ):
          print("feat {} \t importance {:.2f}  ".format(   c,i))


    '''
    LIME
    '''

    features  = df.drop(columns = 'survived').columns

    explainer = lime.lime_tabular.LimeTabularExplainer(
        df[features].astype(int).values,
        mode='classification',
        training_labels=df['Survived'],
        feature_names=model.feature_name())


# asking for explanation for LIME model
i = 1
exp = explainer.explain_instance(df_titanic.loc[i,feat].astype(int).values, prob, num_features=5)


# ---------------
