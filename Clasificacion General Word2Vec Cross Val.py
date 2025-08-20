# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:54:54 2025

@author: esau0
"""
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os


from gensim.models import Word2Vec


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


"""Amazon"""

ruta = r'Amazon_Unlocked_Mobile.csv'
df = pd.read_csv(ruta, encoding= "utf-8")


df['Rating'] = df['Rating'].replace(2, 1) #Ofensivos y Hate 0
df['Rating'] = df['Rating'].replace(4, 0) #Neither 1
df['Rating'] = df['Rating'].replace(5, 0) #Neither 1
df = df[df['Rating'] != 3]

print(df.head())
print(df['Rating'].value_counts())

df = df.dropna()

df, _ = train_test_split(df, train_size=50000, stratify=df['Rating'], random_state=42)


"""Separamos Caracteristicas"""
x = df['Reviews'].astype(str)
y = df['Rating']

print("Shape de x:", len(x))
print("Shape de y:", len(y))
print(df['Rating'].value_counts())





#"""Hate"""
#ruta = r'labeled_data.csv'
#df = pd.read_csv(ruta, encoding= "utf-8") #Para que me adapte todo el texto en espanol, quitar acentos y minusculas
#df = df.dropna()

#df['class'] = df['class'].replace(1, 0) #Ofensivos y Hate 0
#df['class'] = df['class'].replace(2, 1) #Neither 1


#print(df.head())
#print(df['class'].value_counts())

#"""Separamos Caracteristicas"""
#x = df['tweet'].astype(str)
#y = df['class']

#print("Shape de x:", len(x))
#print("Shape de y:", len(y))


#"""Spam"""
#ruta = r'spam_ham_dataset.csv'
#df = pd.read_csv(ruta, encoding= "utf-8")
#df = df[['label', 'text']]

#codificacion = {
#    'ham' : 0, #ham
#   'spam': 1} #spam

#df['label'] = df['label'].map(codificacion)
#df = df.dropna()

#"""Separamos Caracterisitcas"""
#x = df['text'].astype(str)
#y = df['label']



def preprocess_text_w2v(text):
    text = text.lower()
    text = re.sub(r'[^a-zñáéíóúü\s]', '', text)
    words = text.split()
    return words


def ejecutar_experimento_w2v(x, y):
    """
    Función que entrena un modelo Word2Vec y luego evalúa varios clasificadores
    usando validación cruzada estratificada.
    """
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
    
    print("\nPreprocesando texto para Word2Vec...")
    x_train_processed = x_train.apply(preprocess_text_w2v)
    x_test_processed = x_test.apply(preprocess_text_w2v)

    print("Entrenando modelo Word2Vec en el conjunto de entrenamiento...")
    w2v_model = Word2Vec(sentences=x_train_processed, vector_size=100, window=5, min_count=1, workers=10)
    print("Modelo Word2Vec entrenado.")
    
    def document_vector(model, doc):
        doc = [word for word in doc if word in model.wv.index_to_key]
        if not doc:
            return np.zeros(model.vector_size)
        return np.mean(model.wv[doc], axis=0)

    print("Transformando datos de texto a vectores Word2Vec...")
    x_train_vectors = np.array([document_vector(w2v_model, doc) for doc in x_train_processed])
    x_test_vectors = np.array([document_vector(w2v_model, doc) for doc in x_test_processed])
    print("Transformación completada.")


    modelos = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GaussianNB': GaussianNB(), # Se usa GaussianNB para vectores densos
        'MLPClassifier': MLPClassifier(max_iter=500, hidden_layer_sizes=(200, 50), random_state=42)
    }

    if not os.path.exists('graficos_w2v'):
        os.makedirs('graficos_w2v')

    resultados_finales = []

    for nombre, modelo in modelos.items():
        print(f"\n\n======== Procesando Modelo: {nombre} con Validación Cruzada ========\n")
        
        fold_accuracy, fold_precision, fold_recall, fold_f1, fold_auc = [], [], [], [], []

        # VALIDACIÓN CRUZADA ESTRATIFICADA 
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Se itera sobre los vectores de entrenamiento
        for train_index, val_index in kf.split(x_train_vectors, y_train):
            X_train_fold, X_val_fold = x_train_vectors[train_index], x_train_vectors[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            modelo.fit(X_train_fold, y_train_fold)
            y_pred_fold = modelo.predict(X_val_fold)
            
            fold_accuracy.append(accuracy_score(y_val_fold, y_pred_fold))
            fold_precision.append(precision_score(y_val_fold, y_pred_fold, zero_division=0))
            fold_recall.append(recall_score(y_val_fold, y_pred_fold, zero_division=0))
            fold_f1.append(f1_score(y_val_fold, y_pred_fold, zero_division=0))
            
            if hasattr(modelo, "predict_proba"):
                y_probs_fold = modelo.predict_proba(X_val_fold)[:, 1]
                fold_auc.append(roc_auc_score(y_val_fold, y_probs_fold))
            else:
                fold_auc.append(None)
        
        resultados_finales.append({
            'Modelo': nombre,
            'Accuracy': f"{np.mean(fold_accuracy):.4f} ± {np.std(fold_accuracy):.4f}",
            'Precision': f"{np.mean(fold_precision):.4f} ± {np.std(fold_precision):.4f}",
            'Recall': f"{np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}",
            'F1-Score': f"{np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}",
            'AUC': f"{np.mean([auc for auc in fold_auc if auc is not None]):.4f} ± {np.std([auc for auc in fold_auc if auc is not None]):.4f}" if any(fold_auc) else "N/A"
        })
        
        print(f"Generando y guardando gráficos para {nombre} usando el conjunto de prueba final...")
        modelo.fit(x_train_vectors, y_train) 
        y_pred_test = modelo.predict(x_test_vectors)
        
        # Matriz de Confusión
        MC = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 4))
        sns.heatmap(MC, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión Final - {nombre} (Word2Vec)')
        plt.savefig(f'graficos_w2v/matriz_confusion_{nombre}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Curva ROC y Distribución de Probabilidades
        if hasattr(modelo, "predict_proba"):
            y_probs_test = modelo.predict_proba(x_test_vectors)[:, 1]
            auc_test = roc_auc_score(y_test, y_probs_test)
            fpr, tpr, _ = roc_curve(y_test, y_probs_test)
            
            # Gráfico de Curva ROC
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc_test:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='red')
            plt.title(f'Curva ROC Final - {nombre} (Word2Vec)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'graficos_w2v/curva_roc_{nombre}.png', dpi=300, bbox_inches='tight')
            plt.show()
            

            # Gráfico de Distribución de Probabilidades (KDE)
            plt.figure(figsize=(10, 6))
            sns.kdeplot(y_probs_test[y_test == 0], fill=True, label='Clase 0 (Ham)')
            sns.kdeplot(y_probs_test[y_test == 1], fill=True, label='Clase 1 (Spam)')
            plt.title(f"Densidad de Probabilidades - {nombre} (Word2Vec)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'graficos_w2v/distribucion_prob_{nombre}.png', dpi=300, bbox_inches='tight')
            plt.show()
            

    df_resultados = pd.DataFrame(resultados_finales)
    return df_resultados

resultados_w2v_cv = ejecutar_experimento_w2v(x, y)
print("\n\n--- Resultados Promedio (Word2Vec) de la Validación Cruzada (5 Pliegues) ---")
print(resultados_w2v_cv)
