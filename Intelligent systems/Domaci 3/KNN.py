import math

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def dist(l, l_pred):
    return np.mean(np.abs(l - l_pred))


# KNN moj
def distance(row1, row2):
    distance = 0
    for i in range(row1):
        distance += math.pow(row1[i] - row2[i], 2)
    return math.sqrt(distance)


def algorit(x_train, y_train, x_test):
    neigh = int(math.sqrt(len(y_train)))
    distance = np.sqrt(np.sum(np.square(x_train - x_test), axis=1))
    sorted_indexes = np.argsort(distance)
    y_train = y_train[sorted_indexes]
    y_train_nearest = y_train[:neigh]
    return 0 if np.mean(y_train_nearest) < 0.5 else 1


if __name__ == '__main__':
    #Ucitavanje
    pd.set_option('display.max_columns',10)
    pd.set_option('display.width',None)
    data=pd.read_csv("cakes.csv")
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
    output= le.fit_transform(data.type)
    features=data.drop("type", axis=1)
    features=(features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))
    output_features = pd.concat([features,pd.DataFrame(output)], axis=1)
    #Dekartov koordinatni sistem
    column_names = output_features.columns[:-1]
    for c in column_names:
        fig = plt.figure()
        plt.scatter(output_features[-1], output_features[c])
        plt.title(f'Zavisnost type od {c}')
        plt.show()
    corr_1 = output_features.corr()
    labels=np.array(data.type)
    #plt.show()
    #KNN algoritam ugradjen
    f = np.array(features)
    l = np.array(output)
    X_train, X_test, y_train, y_test = train_test_split(f, l, test_size=0.3 , random_state=3)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    Y_predicted = knn.predict(X_test)
    print('Ugradjeni algoritam')
    print(f'Pogresnih primera: {np.sum(np.abs(Y_predicted - y_test))}\nProsecna greska: {np.mean(np.abs(Y_predicted - y_test)) * 100}%')
    print(f'Preciznost modela: {100*(1-np.mean(np.abs(Y_predicted - y_test)))}%\n')
    labels = []
    for x in X_test:
        labels.append(algorit(X_train, y_train, x))

    labels = np.array(labels)
    print('Moj algoritam')
    print(f'Pogresnih primera: {np.sum(np.abs(labels - y_test))}\nProsecna greska: {np.mean(np.abs(labels - y_test)) * 100}%')
    print(f'Preciznost modela: {100*(1-np.mean(np.abs(labels - y_test)))}%')