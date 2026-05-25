# División de datos y modelamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

class modelo_ml():
    def __init__(self, x, y):
        #einicialización
        self.x = x
        self.y = y
    
    def escalar_data (self, x, funcion):
        scaler = funcion()
        X_scaled = scaler.fit_transform(x)
        return X_scaled

    def division_datos(self, tamano, seed_number, estratifica='NO'):
        if (estratifica.upper() == 'NO'):
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=tamano, random_state=seed_number)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=tamano, random_state=seed_number, stratify=self.y)
        return X_train, X_test, y_train, y_test
    
    def matriz_confusion(self, y_test, pred):
        #cnf_matrix = metrics.confusion_matrix(y_test, pred)
        class_names=y_test.unique()#[0,1] #
        cnf_matrix = metrics.confusion_matrix(y_test, pred, labels =class_names)
        #class_names=[0,1] #
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        ax.xaxis.set_label_position("top")
        #tick_marks = np.arange(len(class_names))
        #plt.xticks(tick_marks, class_names)
        #plt.yticks(tick_marks, class_names)
        # create heatmap
        #sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
        #ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label');

    def entrenar_modelo (self, modelo, seed_number, funcion_escalado, estimador='mean_squared_error', tamano=0.2, escalar = 'NO', tipo = 'REGRESION', estratifica='NO'):
        X_train, X_test, y_train, y_test = modelo_ml.division_datos(self, tamano, seed_number, estratifica)
        #Necesitamos escalar?
        if (escalar.upper() == 'SI'):
            scaler = funcion_escalado()
            X_train_escalada = scaler.fit_transform(X_train)
            X_test_escalada = scaler.transform(X_test)
        else:
            X_train_escalada = X_train
            X_test_escalada = X_test

        model_train = modelo.fit(X_train_escalada, y_train)

        y_pred_train =  model_train.predict(X_train_escalada)
        y_pred = model_train.predict(X_test_escalada)

        if (tipo.upper() == 'CLASIFICACION'):
            if estimador == 'accuracy':
                train_score = accuracy_score(y_train, y_pred_train)
                test_score = accuracy_score(y_test, y_pred)
            elif estimador == 'recall':
                train_score = recall_score(y_train, y_pred_train)
                test_score = recall_score(y_test, y_pred)
            elif estimador == 'f1_score':
                train_score = f1_score(y_train, y_pred_train)
                test_score = f1_score(y_test, y_pred)
            elif estimador == 'roc':
                train_score = roc_auc_score(y_train, model_train.predict_proba(X_train_escalada)[:, 1])
                test_score = roc_auc_score(y_test, model_train.predict_proba(X_test_escalada)[:, 1])
            modelo_ml.matriz_confusion(self, y_test, y_pred)
        else:            
            if estimador == 'mean_squared_error':
                train_score = mean_squared_error(y_train, y_pred_train)
                test_score = mean_squared_error(y_test, y_pred)
            elif estimador == 'mean_absolute_error':
                train_score = mean_absolute_error(y_train, y_pred_train)
                test_score = mean_absolute_error(y_test, y_pred)
            elif estimador == 'r2_score':
                train_score = r2_score(y_train, y_pred_train)
                test_score = r2_score(y_test, y_pred)

        print("Entrenamiento estimación : " , train_score)
        print("Entrenamiento score R2 :", model_train.score(X_train_escalada, y_train))
        print("Prueba estimación : ", test_score)
        print("Prueba score R2 :", model_train.score(X_test_escalada, y_test))

        return X_train, X_train_escalada, y_train, X_test, X_test_escalada, y_test, y_pred, train_score, test_score