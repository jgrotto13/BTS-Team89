import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os,sys

current_dir=os.path.abspath(os.path.dirname(sys.argv[0]))

'''GET TRAINING DATA'''
ANALYZED_FEATURES = [x + "_MATCHUP" for x in ["AB", "H", "XBH", "HR", "K", "RBI", "AVG","SLG"]] #+ ["BALLPARK_LABEL"]
ANALYZED_FEATURES.append('BALLPARK_LABEL')
full_dataset = pd.read_csv(current_dir+"/../data/features.csv", index_col="ID")
label_encoder = LabelEncoder()
full_dataset["BALLPARK_LABEL"] = label_encoder.fit_transform(full_dataset["BALLPARK"])

x = full_dataset[ANALYZED_FEATURES]
Y = full_dataset["RESULT"]

# setting a seed just for reproducibility
random_state = 11

x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, train_size=0.80, shuffle=False, random_state=random_state)

''' BUILD MODELS '''
rfr = RandomForestRegressor(n_estimators=400, max_depth=5, min_samples_leaf=20, random_state=random_state).fit(x_train, y_train)

y_pred_train = rfr.predict(x_train).round()
y_pred = rfr.predict(x_test).round()
print('Random Forest - Training Accuracy: {}'.format(accuracy_score(y_train, y_pred_train)))
print('Random Forest - Testing Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

lr = LogisticRegression(random_state=random_state, solver='lbfgs', multi_class='ovr', max_iter=200).fit(x_train, y_train)

y_pred_train_lr = lr.predict(x_train)#.round()
y_pred_lr = lr.predict(x_test)#.round()
print('Logistic Regression - Training Accuracy: {}'.format(accuracy_score(y_train, y_pred_train_lr)))
print('Logistic Regression - Testing Accuracy: {}'.format(accuracy_score(y_test, y_pred_lr)))

''' TUNE PARAMETERS '''
# feature importance
#importances = rfr.feature_importances_
#indices = np.argsort(importances)[::-1]
#for f in range(x_train.shape[1]):
#    print("%s (%f)" % (x_train.columns[indices[f]], importances[indices[f]]))
    
# Hyperparameter tuning
#param_grid = {
#        'n_estimators': [400, 500, 600],
#        #'max_depth': [1, 5, 10],
#        #'criterion': ['gini', 'entropy']
#        'min_samples_leaf': [10, 15, 20]
#}
#
#CV_clf = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=10, verbose=2)
#CV_clf.fit(x_train, y_train)
#print ('Random Forest - Best Parameters: {}'.format(CV_clf.best_params_))
#print ('Random Forest - Best Score: {}'.format(CV_clf.best_score_))

#param_grid = {
#        #'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
#        #'multi_class': ['ovr', 'multinomial', 'auto'],
#        'max_iter': [100, 200, 300]
#}
#
#CV_lr = GridSearchCV(estimator=lr, param_grid=param_grid, cv=10, verbose=2)
#CV_lr.fit(x_train, y_train)
#print ('Logistic Regression - Best Parameters: {}'.format(CV_lr.best_params_))
#print ('Logistic Regression - Best Score: {}'.format(CV_lr.best_score_))
        
''' MAKE PREDICTIONS '''
PREDICTION_FEATURES = [x for x in ["AB", "H", "XBH", "HR", "K", "RBI", "AVG", "SLG", "BALLPARK_LABEL"]]
new_dataset = pd.read_csv(current_dir+"/../data/hot-batters.csv", index_col="Name")
new_dataset["BALLPARK"] = np.where(new_dataset["Home?"]=='Yes', new_dataset["Team"], new_dataset["Opp"].str.replace("@", ""))
new_dataset["BALLPARK"] = new_dataset["BALLPARK"].replace({'WAS': 'WSN', 'TB': 'TBR', 'KC': 'KCR', 'SD': 'SDP', 'CWS': 'CHW', 'SF': 'SFG'})
new_dataset["BALLPARK_LABEL"] = [label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1 for x in new_dataset["BALLPARK"]]

x_new = new_dataset[PREDICTION_FEATURES]

y_new = rfr.predict(x_new)
rfr_prob = np.empty(shape=[0,2])
for i in range(len(x_new)):
    rfr_prob = np.append(rfr_prob, [[x_new.index[i], round(y_new[i], 3)]], axis=0)
    
rfr_df = pd.DataFrame(rfr_prob, columns=['Player', 'Probability'])
rfr_df = rfr_df.sort_values(by=['Probability'], ascending=False)
rfr_df = rfr_df.astype({'Player': str, 'Probability': float})
print("\nRandom Forest: ")
print("Top 5:")
print(rfr_df[:5].values[:,:2].tolist())
prop = round((len(rfr_df) * .2))
if (prop > 5): prop = 5
rfr_df = rfr_df[:prop]
print("\nQualifiers: ")
print(rfr_df.values[:,:2].tolist())

y_new = lr.predict(x_new)
y_proba = lr.predict_proba(x_new)
lr_prob = np.empty(shape=[0,2])
for i in range(len(x_new)):
    lr_prob = np.append(lr_prob, [[x_new.index[i], round(y_proba[i,1], 3)]], axis=0)
    
lr_df = pd.DataFrame(lr_prob, columns=['Player', 'Probability'])
lr_df = lr_df.sort_values(by=['Probability'], ascending=False)
lr_df = lr_df.astype({'Player': str, 'Probability': float})
print("\nLogistic Regression: ")
print("Top 5:")
print(lr_df[:5].values[:,:2].tolist())
lr_df = lr_df[:prop]
print("\nQualifiers: ")
print(lr_df.values[:,:2].tolist())

lr_df['Count'] = lr_df['Player'].map(rfr_df['Player'].value_counts())
rfr_df['Count'] = rfr_df['Player'].map(lr_df['Player'].value_counts())
df_pick = lr_df.loc[lr_df['Count'] == 1.0]
df_tie = rfr_df.loc[rfr_df['Count'] == 1.0]
df_tiebreaker = pd.concat([df_pick, df_tie], axis=1)
df_tiebreaker['Mean'] = df_tiebreaker[['Probability', 'Probability']].mean(axis=1)
df_tiebreaker = df_tiebreaker.sort_values(by=['Mean'], ascending=False)

''' MAKE PICK '''
if (len(lr_df.loc[lr_df['Count'] == 1.0]) == 1):
    if (df_tiebreaker.iloc[0,6] >= .285):
        print('\nTodays Pick: ' + df_pick.iloc[0,0])
    else:
        print('\nNo Pick Today...')
elif (len(lr_df.loc[lr_df['Count'] == 1.0]) == 2):
    if (df_tiebreaker.iloc[0,6] >= .285 and df_tiebreaker.iloc[1,6] >= .285):
        print('\nTodays Pick: Double Down!')
        print(df_pick.iloc[0,0] + ' & ' + df_pick.iloc[1,0])
    else:
        print('\nNo Pick Today...')
elif (len(lr_df.loc[lr_df['Count'] == 1.0]) > 2):
    if (df_tiebreaker.iloc[0,6] >= .285 and df_tiebreaker.iloc[1,6] >= .285):
        print('\nTodays Pick: Double Down!')
        print(df_tiebreaker.iloc[0,0] + ' & ' + df_tiebreaker.iloc[1,0])
    else:
        print('\nNo Pick Today...')
else:
    print('\nNo Pick Today...')
