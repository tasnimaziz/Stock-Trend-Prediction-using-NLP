import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv("Sent_scores.csv", index_col=0)
df["PosCount_cv"] = df["PosCount"].rolling(window=5).sum()
df["NegCount_cv"] = df["NegCount"].rolling(window=5).sum()
df["TrustCount_cv"] = df["TrustCount"].rolling(window=5).sum()
df["AngerCount_cv"] = df["AngerCount"].rolling(window=5).sum()
df["AnticipationCount_cv"] = df["AnticipationCount"].rolling(window=5).sum()
df["DisgustCount_cv"] = df["DisgustCount"].rolling(window=5).sum()
df["FearCount_cv"] = df["FearCount"].rolling(window=5).sum()
df["JoyCount_cv"] = df["JoyCount"].rolling(window=5).sum()
df["SadnessCount_cv"] = df["SadnessCount"].rolling(window=5).sum()
df["SurpriseCount_cv"] = df["SurpriseCount"].rolling(window=5).sum()

stock_df = pd.read_csv("DJIA_table.csv", index_col=1)
stock_df = stock_df.iloc[::-1]

df1 = pd.merge(df, stock_df)
df2 = df1.drop('Volume', axis=1)
df2 = df2.drop('High', axis=1)
df2 = df2.drop('Low', axis=1)
#df2 = df2.drop('Close', axis=1)
#df2 = df2.drop('Date', axis=1)
for i in range(1,len(df2["Adj Close"])):
    if(df2["Adj Close"][i]>=df2["Adj Close"][i-1]):
        df2["Close"][i]=1
    else:
        df2["Close"][i]=0

df2 = df2.drop(df2.index[[0]])
change = df2["Close"]
df2 = df2.drop('Close', axis=1)
train = df2.loc[0:1392,:]
test = df2.drop(train.index)
date_train = train["Date"]
date_test = test["Date"]
train = train.drop('Date', axis=1)
test = test.drop('Date',axis=1)
X = train.loc[:, train.columns!="Adj Close"]
y = train.iloc[:,-1]
test_X = test.loc[:, test.columns!="Adj Close"]
test_y = test.iloc[:,-1]

#clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = GradientBoostingRegressor(n_estimators=1000, max_depth=10)
#clf = linear_model.LinearRegression()

clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test["change"] = 0
test["change_predict"] = 0
predict = test["predict"]

for i in range(1,len(test["Adj Close"])):
    if(test["Adj Close"].iloc[i]>=test["Adj Close"].iloc[i-1]):
        test["change"].iloc[i]=1
    else:
        test["change"].iloc[i]=0

for i in range(1,len(test["predict"])):
    if(test["predict"].iloc[i]>=test["predict"].iloc[i-1]):
        test["change_predict"].iloc[i]=1
    else:
        test["change_predict"].iloc[i]=0
        
        
#graph
x = np.arange(596)

plt.plot(x, test["change"])
plt.plot(x, test["change_predict"])

#plt.legend(['y = x', 'y = 2x'], loc='upper left')

plt.show()

#confusion_matrix
cnf_matrix = confusion_matrix(test["change"], test["change_predict"])
np.set_printoptions(precision=2)
class_names = ['1','0']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

print(accuracy_score(test["change"], test["change_predict"]))
print(confusion_matrix(test["change"], test["change_predict"]))
print(classification_report(test["change"], test["change_predict"]))