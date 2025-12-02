# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:54:54 2025
Refactored on Sat Jun 15 01:30:00 2025

@author: esau0
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

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


from transformers import BertTokenizer, BertModel
import torch

import psutil
import platform
from memory_profiler import memory_usage

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
    
# Con que dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model_bert = BertModel.from_pretrained(MODEL_NAME).to(device) # Mueve el modelo a la GPU/CPU

#"""Amazon"""

#ruta = r'C:\Users\esau0\Desktop\Maestria D\Materias\Machine Learning\Vectorización\Amazon_Unlocked_Mobile.csv'
#df = pd.read_csv(ruta, encoding= "utf-8") #Para que me adapte todo el texto en espanol, quitar acentos y minusculas


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
ruta = r'C:\Users\esau0\Desktop\Maestria D\Materias\Machine Learning\Vectorización\labeled_data.csv'
df = pd.read_csv(ruta, encoding= "utf-8") #Para que me adapte todo el texto en espanol, quitar acentos y minusculas
df = df.dropna()

df['class'] = df['class'].replace(1, 0) #Ofensivos y Hate 0
df['class'] = df['class'].replace(2, 1) #Neither 1


print(df.head())
print(df['class'].value_counts())

"""Separamos Caracteristicas"""
x = df['tweet'].astype(str) 
y = df['class']

print("Shape de x:", len(x))
print("Shape de y:", len(y))


#"""Spam"""
#ruta = r'C:\Users\esau0\Desktop\Maestria D\Materias\Machine Learning\Vectorización\spam_ham_dataset.csv'
#df = pd.read_csv(ruta, encoding= "utf-8")
#df = df[['label', 'text']]

#codificacion = {
 #   'ham' : 0, #ham
  # 'spam': 1} #spam
#df['label'] = df['label'].map(codificacion)
#df = df.dropna()
#"""Separamos Caracterisitcas"""
#x = df['text'].astype(str)
#y = df['label']



def extract_bert_embeddings(texts, tokenizer, model, batch_size=32, max_length=128):
    model.eval()
    embeddings = []
    
    # texts como lista
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
        
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            output = model(**encoded_input)
        
        cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)
        
    return np.vstack(embeddings)

def ejecutar_experimento_bert_cv(x, y):
    """
    Función que extrae embeddings de BERT y evalúa varios clasificadores
    usando validación cruzada estratificada.
    """

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
    


    print("\nExtrayendo embeddings de BERT para datos de entrenamiento...")
    inicio_vec = time.time()  
    
    x_train_vectors = extract_bert_embeddings(x_train, tokenizer, model_bert)
    print(f"Dimensiones de embeddings de entrenamiento: {x_train_vectors.shape}")

    print("\nExtrayendo embeddings de BERT para datos de prueba...")
    x_test_vectors = extract_bert_embeddings(x_test, tokenizer, model_bert)
    print(f"Dimensiones de embeddings de prueba: {x_test_vectors.shape}")
    
    fin_vec = time.time()
    tiempo_vectorizacion = fin_vec - inicio_vec  
    
    
    
    
    modelos = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GaussianNB': GaussianNB(),
        'MLPClassifier': MLPClassifier(max_iter=500, hidden_layer_sizes=(200, 50), activation="relu", random_state=42)
    }

    if not os.path.exists('graficos_bert_cv'):
        os.makedirs('graficos_bert_cv')

    resultados_finales = []

    for nombre, modelo in modelos.items():
        print(f"\n\n======== Procesando Modelo: {nombre} con Validación Cruzada ========\n")
        
       # ======== TIEMPO SOLO DEL MODELO (NO BERT) ========
        inicio_modelo = time.time()  
        mem_inicio = psutil.Process().memory_info().rss / (1024 ** 2)
        
        

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        

        fold_accuracy, fold_precision, fold_recall, fold_f1, fold_auc = [], [], [], [], []

        for fold, (train_index, val_index) in enumerate(kf.split(x_train_vectors, y_train)):
            print(f"--- Fold {fold+1}/5 ---")
            
 
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
                fold_auc.append(float('nan')) # Usar NaN si no se puede calcular AUC
        
        
       
        
        
        print(f"\nGenerando y guardando gráficos para {nombre} usando el conjunto de prueba final...")

        modelo.fit(x_train_vectors, y_train)
        y_pred_test = modelo.predict(x_test_vectors)
        
        
        
        #Tiempo
        fin_modelo = time.time()
        tiempo_modelo = fin_modelo - inicio_modelo  
        mem_fin = psutil.Process().memory_info().rss / (1024 ** 2)
        memoria_usada = max(mem_fin - mem_inicio, 0)

      
        tiempo_total = tiempo_vectorizacion + tiempo_modelo 

        resultados_finales.append({
            'Modelo': nombre,
            'Accuracy': f"{np.mean(fold_accuracy):.4f} ± {np.std(fold_accuracy):.4f}",
            'Precision': f"{np.mean(fold_precision):.4f} ± {np.std(fold_precision):.4f}",
            'Recall': f"{np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}",
            'F1-Score': f"{np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}",
            'AUC': f"{np.nanmean(fold_auc):.4f} ± {np.nanstd(fold_auc):.4f}" if not np.isnan(fold_auc).all() else "N/A",
            'Runtime (s)': round(tiempo_total, 2),   
            'Memory (MiB)': round(memoria_usada, 1)
        })
        
        
        
        MC = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 4))
        sns.heatmap(MC, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
        plt.title(f'Confusion Matrix - {nombre} (BERT)')
        plt.savefig(f'graficos_bert_cv/matriz_confusion_{nombre}.png', dpi=350, bbox_inches='tight')
        plt.show()

        # Curva ROC y Distribución de Probabilidades
        if hasattr(modelo, "predict_proba"):
            y_probs_test = modelo.predict_proba(x_test_vectors)[:, 1]
            auc_test = roc_auc_score(y_test, y_probs_test)
            fpr, tpr, _ = roc_curve(y_test, y_probs_test)
            
            # Gráfico de Curva ROC
            plt.figure(figsize=(10, 7))
            plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc_test:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='red')
            plt.title(f'ROC Curve - {nombre} (BERT)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'graficos_bert_cv/curva_roc_{nombre}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Gráfico de Distribución de Probabilidades
            plt.figure(figsize=(10, 7))
            sns.kdeplot(y_probs_test[y_test == 0], fill=True, label='Class 0')
            sns.kdeplot(y_probs_test[y_test == 1], fill=True, label='Class 1')
            plt.title(f"Probability Density - {nombre} (BERT)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'graficos_bert_cv/distribucion_prob_{nombre}.png', dpi=350, bbox_inches='tight')
            plt.show()
            
    df_resultados = pd.DataFrame(resultados_finales)
    return df_resultados

resultados_bert_cv = ejecutar_experimento_bert_cv(x, y)
print("\n\n--- Resultados Promedio (BERT) de la Validación Cruzada (5 Pliegues) ---")
print(resultados_bert_cv)