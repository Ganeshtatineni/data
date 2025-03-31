import pandas as pd

file_path = "iris.csv"  
df = pd.read_csv(file_path)

y = df['Species']
x = df.drop('Species', axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

y_train_rf_pred = rf.predict(x_train)
y_test_rf_pred = rf.predict(x_test)

from sklearn.metrics import  accuracy_score
y_train_as=accuracy_score(y_train,y_train_rf_pred)
y_test_as=accuracy_score(y_test,y_test_rf_pred)


print('y_train_as=' ,y_train_as)
print('y_test_as=',y_test_as)

import matplotlib.pyplot as plt
plt.scatter(x=y_test,y=y_test_rf_pred)
plt.show()


print(x_test, y_test_rf_pred)

















