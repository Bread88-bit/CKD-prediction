from pyscript import document

result_placeholder = ["#result_one_d","#result_one_i","#result_two_d","#result_two_i"]
output_div = []
for l in result_placeholder:
  output_div.append(document.querySelector(l))


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone

def submitbtn(event):
    data = {
    "age": [document.querySelector("#age").value],
    "hypertension" : [int(document.querySelector('input[name="hypertension"]:checked').value)],
    "diabetes mellitus" : [int(document.querySelector('input[name="diabetes"]:checked').value)],
    "coronary artery disease" : [int(document.querySelector('input[name="cad"]:checked').value)],
    "appetite" : [int(document.querySelector("#appetite").value)],
    "anemia" : [int(document.querySelector('input[name="anemia"]:checked').value)],
    "pedal edema" : [int(document.querySelector('input[name="pedal_edema"]:checked').value)]
    }

    df_user = pd.DataFrame(data)

    df = pd.read_csv("df_filled.csv")


    scaler=StandardScaler()
    df['age']=scaler.fit_transform(df[['age']])
    df_user['age']=scaler.transform(df_user[['age']])
    X=df.drop(['ckd or not ckd'],axis=1)
    y=df['ckd or not ckd']

    models = {
    'RandomForestClassifier': RandomForestClassifier(
        bootstrap=True,
        class_weight='balanced',
        criterion='gini',
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=100,
        random_state=42
    ),
    'ExtraTreesClassifier': ExtraTreesClassifier(
        bootstrap=True,
        class_weight='balanced',
        criterion='gini',
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=100,
        random_state=42
    ),
    'GradientBoostingClassifier': GradientBoostingClassifier(
        criterion='friedman_mse',
        learning_rate=0.01,
        max_depth=5,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=100,
        subsample=0.8,
        random_state=42
    ),
    'AdaBoostClassifier': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        learning_rate=1.0,
        n_estimators=100,
        random_state=42
    ),
    'GaussianNB': GaussianNB(
        var_smoothing=1e-11
    ),
    'LogisticRegression': LogisticRegression(
        C=1,
        class_weight='balanced',
        max_iter=1000,
        penalty='l2',
        solver='lbfgs',
        random_state=42
    ),
    'DecisionTreeClassifier': DecisionTreeClassifier(
        ccp_alpha=0.0,
        class_weight='balanced',
        criterion='gini',
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=2,
        splitter='best',
        random_state=42
    ),
    'SVC': SVC(
        C=10,
        class_weight='balanced',
        degree=2,
        gamma='scale',
        kernel='rbf',
        probability=True,
        random_state=42
    ),
    'XGBClassifier': XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=1,
        n_estimators=100,
        objective='binary:logistic',
        subsample=0.8,
        tree_method='auto',
        random_state=42,
        eval_metric='logloss'
    ),
    'KNeighborsClassifier': KNeighborsClassifier(
        algorithm='auto',
        leaf_size=30,
        n_neighbors=3,
        weights='uniform'
    )
    }

    tier1_clf = VotingClassifier(
        estimators=[('GaussianNB', models['GaussianNB']), ('LogisticRegression', models['LogisticRegression'])],
        voting='soft'  # Weighted probability average
    )

    tier1_clf.fit(X, y)
    result1 = [tier1_clf.predict(df_user),tier1_clf.predict_proba(df_user)]
    #print("High Sensitivity Voting Classifier\nDiagnosis:",result1[0],"\nIndex:",result1[1].reshape(-1)[1])
    output_div[0].innerHTML = (result1[0] == float(1))
    output_div[1].innerHTML = result1[1].reshape(-1)[1]
    tier2_clf = VotingClassifier(
        estimators=[(name, clone(model)) for name, model in models.items()],
        voting='soft'
    )

    tier2_clf.fit(X, y)
    result2 = [tier2_clf.predict(df_user),tier2_clf.predict_proba(df_user)]
    #print("Full Ensemble Voting Classifier\nDiagnosis:",result2[0],"\nIndex:",result2[1].reshape(-1)[1])
    output_div[2].innerHTML = (result2[0] == float(1))
    output_div[3].innerHTML = result2[1].reshape(-1)[1]

    return "done"