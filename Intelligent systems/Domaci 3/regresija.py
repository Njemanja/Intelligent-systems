import numpy as np
from sklearn.linear_model import LinearRegression
import math
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class MyLinearRegression:
    def __init__(self, cols_num, lr=0.0001, iter_num=1000):
        self.theta = np.zeros([1, cols_num + 1])
        self.lr = lr
        self.iter_num = iter_num
        self.mse=[]
    def train(self, x_train, y_train):
        x_train = self.expand_features(x_train)
        for i in range(self.iter_num):
            grad = ((self.theta@ x_train.T).T - y_train).T @ x_train / len(y_train)
            # loss = np.mean(np.square((y_train - self.theta.T*x_train)))
            # self.mse.append(loss)
            ### plt.plot(loss)
            self.theta = self.theta - self.lr * grad

    def predict(self, x_test):
        return (self.theta @ self.expand_features(x_test).T).T

    @staticmethod
    def expand_features(x_train):
        ones = np.ones((len(x_train), 1))
        return np.concatenate((ones, x_train), axis=1)

def mean_sqrt_error(y1, y2):
    return np.mean(np.sqrt(np.sum(np.square(y1 - y2), axis=1)))


 #Ucitavanje
pd.set_option('display.max_columns',20)
pd.set_option('display.width',None)
data=pd.read_csv("fuel_consumption.csv")
#Prvih 5
print(data.head())
#Informacije
print(data.info())
print(data.describe())
print(data.describe(include=[object]))
#Korelaciona matrica
korelacionaMatrica = data.corr()
sb.heatmap(korelacionaMatrica, annot=True)
#Transformaicja izlaza
le= LabelEncoder()
data.FUELTYPE= le.fit_transform(data.FUELTYPE)
data.MODELYEAR= le.fit_transform(data.MODELYEAR)
data.MAKE= le.fit_transform(data.MAKE)
data.MODEL= le.fit_transform(data.MODEL)
data.VEHICLECLASS= le.fit_transform(data.VEHICLECLASS)
data.ENGINESIZE= le.fit_transform(data.ENGINESIZE)
data.CYLINDERS= le.fit_transform(data.CYLINDERS)
data.TRANSMISSION= le.fit_transform(data.TRANSMISSION)
#Brisanje/Popunjavanje praznih
data.FUELTYPE=data.FUELTYPE.fillna(data.FUELTYPE.mean())
data.FUELCONSUMPTION_CITY=data.FUELCONSUMPTION_CITY.fillna(data.FUELCONSUMPTION_CITY.mean())
data.FUELCONSUMPTION_HWY=data.FUELCONSUMPTION_HWY.fillna(data.FUELCONSUMPTION_HWY.mean())
data.FUELCONSUMPTION_COMB=data.FUELCONSUMPTION_COMB.fillna(data.FUELCONSUMPTION_COMB.mean())
data.FUELCONSUMPTION_COMB_MPG=data.FUELCONSUMPTION_COMB_MPG.fillna(data.FUELCONSUMPTION_COMB_MPG.mean())


output=data.CO2EMISSIONS
features=data.drop("CO2EMISSIONS", axis=1)
cor = []
f=np.array(features)
l=np.array(output)
for i in range(f.shape[1]):
    cor.append(np.correlate((f[:, i] - np.mean(f[:, i]))/np.std(f[:, i]), (l - np.mean(l))/np.std(l)))
fig1 = plt.figure()
plt.xticks(rotation=45)
plt.bar(np.array(["Model year","MAKE","MODEL","VEHICLECLASS"
                     ,"ENGINESIZE","CYLINDERS","TRANSMISSION",
                  "FUELTYPE","FUELCONSUMPTION_CITY",
                  "FUELCONSUMPTION_HWY",
                  "FUELCONSUMPTION_COMB",
                  "FUELCONSUMPTION_COMB_MPG"]),np.squeeze(np.array(cor)))
plt.show()
features=features.drop("MODELYEAR", axis=1)
features=features.drop("MAKE", axis=1)
features=features.drop("MODEL", axis=1)
features=features.drop("TRANSMISSION", axis=1)
features=features.drop("FUELTYPE", axis=1)

output_features = pd.concat([features,pd.DataFrame(output)], axis=1)
#Dekartov koordinatni sistem
column_names = output_features.columns[1:]
for c in column_names:
    fig = plt.figure()
    plt.scatter(output_features["CO2EMISSIONS"], output_features[c])
    plt.title(f'Zavisnost CO2EMISSIONS od {c}')
    plt.show()
corr_1 = output_features.corr()
labels=np.array(data.CO2EMISSIONS)


#plt.show()

f = np.array(features)
l = np.array(output)
x_train, x_test, y_train, y_test = train_test_split(f, l, test_size=0.3, random_state=3)

"""cor = []
for i in range(f.shape[1]):
    cor.append(np.correlate((f[:, i] - np.mean(f[:, i]))/np.std(f[:, i]), (l - np.mean(l))/np.std(l)))
fig1 = plt.figure()
plt.bar(np.array([1,2,3,4,5]),np.squeeze(np.array(cor)))
plt.show()"""


y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))
lin = MyLinearRegression(x_train.shape[1])
lin.train(x_train, y_train)
#plt.plot(lin.mse,'')
y_pred_my = lin.predict(x_test)
print(f'MY RMSE: {mean_sqrt_error(y_pred_my,y_test)}')
print(f'Theta: {lin.theta[0, 1:]}')
# y = (6, 1) y = np.reshape(y, (-1, 1))
lr_model = LinearRegression()
lr_model.fit(MyLinearRegression.expand_features(x_train), y_train)

y_pred = lr_model.predict(MyLinearRegression.expand_features(x_test))
print(f'RMSE: {mean_sqrt_error(y_pred,y_test)}')


