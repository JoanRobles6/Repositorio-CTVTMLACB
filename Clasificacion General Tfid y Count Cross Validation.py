# -*- coding: utf-8 -*-
"""
Created on Sun May  4 01:07:23 2025

@author: esau0 

"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import torch
import torch.nn as nn
import torch.optim as optim

import psutil
import platform
from memory_profiler import memory_usage

import random
random.seed(42)
np.random.seed(42)

#
#"""Amazon"""

#ruta = r'C:\Users\esau0\Desktop\Maestria D\Materias\Machine Learning\Vectorización\Amazon_Unlocked_Mobile.csv'
#df = pd.read_csv(ruta, encoding= "utf-8")

#df['Rating'] = df['Rating'].replace(2, 1) #Ofensivos y Hate 0
#df['Rating'] = df['Rating'].replace(4, 0) #Neither 1
#df['Rating'] = df['Rating'].replace(5, 0) #Neither 1
#df = df[df['Rating'] != 3]

#print(df.head())
#print(df['Rating'].value_counts())

#df = df.dropna()

#df, _ = train_test_split(df, train_size=50000, stratify=df['Rating'], random_state=42)


#"""Separamos Caracteristicas"""
#x = df['Reviews'].astype(str)
#y = df['Rating']

#print("Shape de x:", len(x))
#print("Shape de y:", len(y))
#print(df['Rating'].value_counts())



#"""Hate"""
#ruta = r'C:\Users\esau0\Desktop\Maestria D\Materias\Machine Learning\Vectorización\labeled_data.csv'
#df = pd.read_csv(ruta, encoding= "utf-8") 
#df = df.dropna()

#df['class'] = df['class'].replace(1, 0) #Ofensivos y Hate 0
#df['class'] = df['class'].replace(2, 1) #Neither 1


#print(df.head())
#print(df['class'].value_counts())

#Separamos Caracteristicas"""
#x = df['tweet'].astype(str)
#y = df['class']

#print("Shape de x:", len(x))
#print("Shape de y:", len(y))


#"""Spam"""
ruta = r'C:\Users\esau0\Desktop\Maestria D\Materias\Machine Learning\Vectorización\spam_ham_dataset.csv'
df = pd.read_csv(ruta, encoding= "utf-8")
df = df[['label', 'text']]

codificacion = {
    'ham' : 0, #ham
   'spam': 1} #spam

df['label'] = df['label'].map(codificacion)
df = df.dropna()

"""Separamos Caracterisitcas"""
x = df['text'].astype(str)
y = df['label']




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)


print("=== Hardware Information ===")
print(f"CPU: {platform.processor()}")
print(f"RAM total: {round(psutil.virtual_memory().total / 1e9, 2)} GB")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print("=============================\n")


def ejecutar_con_validacion_cruzada(x_train, y_train, x_test, y_test):
    """
    Esta función aplica validación cruzada para evaluar múltiples modelos
    con un vectorizador específico y guarda los gráficos resultantes.
    """
    
    vectorizador = TfidfVectorizer()
    #vectorizador = CountVectorizer()
    
    modelos = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_jobs=-1, random_state=42),
        'MultinomialNB': MultinomialNB(),
        'MLPClassifier': MLPClassifier(max_iter=500, hidden_layer_sizes=(200, 50), activation="relu", random_state=42)
    }

    
    if not os.path.exists('graficos'):
        os.makedirs('graficos')

    resultados_finales = []

    for nombre, modelo in modelos.items():
        print(f"\n\n======== Procesando Modelo: {nombre} ========\n")
        
        inicio_tiempo = time.time()
        mem_inicio = psutil.Process().memory_info().rss / (1024 ** 2)  # memoria inicial (MiB)

        
        fold_accuracy, fold_precision, fold_recall, fold_f1, fold_auc = [], [], [], [], []

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train), start=1):
            print(f"Procesando fold {i} de 5 para el modelo {nombre}")
            X_train_fold, X_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            X_train_fold_vec = vectorizador.fit_transform(X_train_fold)
            X_val_fold_vec = vectorizador.transform(X_val_fold)

            modelo.fit(X_train_fold_vec, y_train_fold)
            y_pred_fold = modelo.predict(X_val_fold_vec)
            
            fold_accuracy.append(accuracy_score(y_val_fold, y_pred_fold))
            fold_precision.append(precision_score(y_val_fold, y_pred_fold, zero_division=0))
            fold_recall.append(recall_score(y_val_fold, y_pred_fold, zero_division=0))
            fold_f1.append(f1_score(y_val_fold, y_pred_fold, zero_division=0))
            
            if hasattr(modelo, "predict_proba"):
                y_probs_fold = modelo.predict_proba(X_val_fold_vec)[:, 1]
                fold_auc.append(roc_auc_score(y_val_fold, y_probs_fold))
            else:
                fold_auc.append(None)
        
        
        x_train_vec_full = vectorizador.fit_transform(x_train)
        x_test_vec_full = vectorizador.transform(x_test)
        
        modelo.fit(x_train_vec_full, y_train)
        y_pred_test = modelo.predict(x_test_vec_full)
        
        #Guardamos la medicion
        fin_tiempo = time.time()
        mem_fin = psutil.Process().memory_info().rss / (1024 ** 2)
        tiempo_total = fin_tiempo - inicio_tiempo
        memoria_usada = max(mem_fin - mem_inicio, 0)
        print(f"\nTiempo total: {tiempo_total:.2f} segundos")
        print(f"Memoria usada: {memoria_usada:.1f} MiB\n")
        
        resultados_finales.append({
            'Modelo': nombre,
            'Accuracy': f"{np.mean(fold_accuracy):.4f} ± {np.std(fold_accuracy):.4f}",
            'Precision': f"{np.mean(fold_precision):.4f} ± {np.std(fold_precision):.4f}",
            'Recall': f"{np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}",
            'F1-Score': f"{np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}",
            'AUC': f"{np.mean([auc for auc in fold_auc if auc is not None]):.4f} ± {np.std([auc for auc in fold_auc if auc is not None]):.4f}" if any(fold_auc) else "N/A",
            'Runtime (s)': round(tiempo_total, 2),
            'Memory (MiB)': round(memoria_usada, 1)
        })

        
        print(f"Generando gráficos para {nombre}...")
       
                # === MATRIZ DE CONFUSIÓN ===
        MC = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(10, 7))
        sns.heatmap(MC, annot=True, fmt='d', cmap='Blues', cbar=False,
                    annot_kws={"size": 14})
        plt.title(f'Confusion Matrix - {nombre}', fontsize=16, weight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'graficos/matriz_confusion_{nombre}.png', dpi=350, bbox_inches='tight')
        plt.show()
        
        
        # === CURVA ROC ===
        if hasattr(modelo, "predict_proba"):
            y_probs_test = modelo.predict_proba(x_test_vec_full)[:, 1]
            auc_test = roc_auc_score(y_test, y_probs_test)
            fpr, tpr, _ = roc_curve(y_test, y_probs_test)
        
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {auc_test:.3f})', color='navy')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.5)
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title(f'ROC Curve - {nombre}', fontsize=16, weight='bold')
            plt.legend(fontsize=12, loc='lower right')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'graficos/curva_roc_{nombre}.png', dpi=350, bbox_inches='tight')
            plt.show()
        
        
            # === DISTRIBUCIÓN DE PROBABILIDADES ===
            plt.figure(figsize=(10, 7))
            sns.kdeplot(y_probs_test[y_test == 0], color='#1f77b4', fill=True, alpha=0.5, label='Class 0 ')
            sns.kdeplot(y_probs_test[y_test == 1], color='#d62728', fill=True, alpha=0.5, label='Class 1 ')
            plt.xlabel("Predicted Probability", fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.title(f"Predicted Probability Distribution - {nombre}", fontsize=16, weight='bold')
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'graficos/distribucion_probabilidades_{nombre}.png', dpi=350, bbox_inches='tight')
            plt.show()
                
            

    df_resultados = pd.DataFrame(resultados_finales)
    return df_resultados

#Resultados
resultados_cv = ejecutar_con_validacion_cruzada(x_train, y_train, x_test, y_test)
print("\n\n--- Resultados Promedio de la Validación Cruzada (5 Pliegues) ---")
print(resultados_cv)