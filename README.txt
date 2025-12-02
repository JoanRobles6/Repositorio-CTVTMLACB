# [Comparision of Text Vectorization Techniques for Machine Learning Applied to Binary Classification]

This repository contains the source code, datasets, and experimental results associated with the research paper titled **"[Comparision of Text Vectorization Techniques for Machine Learning Applied to Binary Classification]"**.

The project performs a comparative analysis of text vectorization techniques (BoW, TF-IDF, Word2Vec, Doc2Vec, BERT, among others) applied to classification tasks, along with their statistical validation.

## 📂 Repository Structure

The files are organized as follows:

- **`Dataset.zip`**: Compressed archive containing the 3 databases used in the experiments.
- **`Resultados Modelos - Cross Val Rev.xlsx`**: Excel file containing detailed data from the experiments.
- **Python Scripts** (Root directory): Scripts for the machine learning models (detailed below).
- **R Scripts**: `Shapiro.R` and `U - Test.R` for hypothesis testing.
- **`Imagenes/`**: Folder containing all resulting charts and plots.

---

## 🚀 Execution Instructions (Python)

The classification and vectorization experiments are distributed across separate Python scripts to facilitate execution.

### Script Organization
1. **BoW and TF-IDF**: Both classical methods are unified within the script:
   - `Clasificacion General Tfid y Count Cross Validation.py`
2. **Embeddings and Advanced Models**: These have their own individual scripts:
   - Word2Vec: `Clasificacion General Word2Vec Cross Val.py`
   - Doc2Vec: `Clasificacion General Doc2 Vec Cross Val.py`
   - BERT: `Clasificacion General Transformers Bert Cross Val.py`

### ⚠️ Dataset Configuration (Preprocessing)
Included within each Python script is the preprocessing code for the **3 databases** used in this study.

> **Important Note:** The data loading and preprocessing lines are commented out by default. To run a specific experiment:
> 1. Open the desired script.
> 2. Locate the data loading section.
> 3. **Uncomment** the lines corresponding to the database you wish to process and ensure the other two remain commented out.

---

## 📊 Statistical Analysis (R)

Two scripts written in R are included to statistically validate the results:

1. **`Shapiro.R`**: Performs the Shapiro-Wilk test to check for data normality.
2. **`U - Test.R`**: Performs the Mann-Whitney U Test (U-Test) for the remaining non-parametric comparisons.

---

## 📈 Results

### Excel File (`Resultados Modelos - Cross Val Rev.xlsx`)
This file consolidates all quantitative information:
* **Sheet 1 (General):** Complete table with raw results from all experiments.
* **Sheet 2 (Analysis & Treatment):** Treated data, calculation of statistical measures (such as the mean), and specific results of the **T-Test** performed between BoW and TF-IDF.

### Images
All charts and visualizations generated during the experiments are stored within the general `Imagenes/` folder.

---

## 📦 Installation and Requirements

To run the code, please ensure you unzip `Dataset.zip` in the project root.

Recommended setup:
* Python 3.x
* Key libraries: `pandas`, `numpy`, `sklearn`, `tensorflow/torch` (as required for BERT), `gensim` (for Word2Vec/Doc2Vec).
* R (for the statistical scripts).

---

## 📝 Citation and Contact

If you use this code or the results for your research, please cite the corresponding paper or contact the author via this repository.




# [Comparación de técnicas de vectorización para machine learning aplicado a clasificación binaria]

Este repositorio contiene el código fuente, los datasets y los resultados experimentales asociados al artículo de investigación titulado **"[Comparision of Text Vectorization Techniques for Machine Learning Applied to Binary Classification]"**.

El proyecto realiza un análisis comparativo de técnicas de vectorización de texto (BoW, TF-IDF, Word2Vec, Doc2Vec, BERT, entre otros) aplicadas a tareas de clasificación, junto con su validación estadística.

## 📂 Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

- **`Dataset.zip`**: Archivo comprimido que contiene las 3 bases de datos utilizadas en los experimentos.
- **`Resultados Modelos - Cross Val Rev.xlsx`**: Archivo de Excel con los datos detallados de los experimentos.
- **Scripts de Python** (Raíz): Scripts para los modelos de aprendizaje (detallados más abajo).
- **Scripts de R**: `Shapiro.R` y `U - Test.R` para pruebas de hipótesis.
- **`Imagenes/`**: Carpeta que contiene todas las gráficas resultantes.

---

## 🚀 Instrucciones de Ejecución (Python)

Los experimentos de clasificación y vectorización se encuentran en scripts de Python separados para facilitar su ejecución.

### Organización de los Scripts
1. **BoW y TF-IDF**: Ambos métodos clásicos se encuentran unificados en el script:
   - `Clasificacion General Tfid y Count Cross Validation.py`
2. **Embeddings y Modelos Avanzados**: Tienen sus propios scripts individuales:
   - Word2Vec: `Clasificacion General Word2Vec Cross Val.py`
   - Doc2Vec: `Clasificacion General Doc2 Vec Cross Val.py`
   - BERT: `Clasificacion General Transformers Bert Cross Val.py`

### ⚠️ Configuración de Datasets (Preprocesamiento)
Dentro de cada script de Python, se incluye el código de preprocesamiento para las **3 bases de datos** utilizadas en el estudio.

> **Nota Importante:** El código de carga y preprocesamiento está comentado por defecto. Para correr un experimento específico:
> 1. Abre el script deseado.
> 2. Localiza la sección de carga de datos.
> 3. **Descomenta** las líneas correspondientes a la base de datos que deseas procesar y asegúrate de comentar las otras dos.

---

## 📊 Análisis Estadístico (R)

Se incluyen dos scripts en lenguaje R para la validación estadística de los resultados:

1. **`Shapiro.R`**: Script encargado de realizar el test de Shapiro-Wilk para comprobar la normalidad de los datos.
2. **`U - Test.R`**: Script para realizar la prueba U de Mann-Whitney (U-Test) para las comparaciones no paramétricas restantes.

---

## 📈 Resultados

### Archivo Excel (`Resultados Modelos - Cross Val Rev.xlsx`)
Este archivo consolida toda la información cuantitativa:
* **Hoja 1 (General):** Tabla completa con los resultados crudos de todos los experimentos.
* **Hoja 2 (Análisis y Tratamiento):** Datos tratados, cálculo de medidas estadísticas (como la media) y resultados específicos de la **Prueba T** realizada entre BoW y TF-IDF.

### Imágenes
Todas las gráficas y visualizaciones generadas durante los experimentos se encuentran almacenadas en la carpeta `Imagenes/`.

---

## 📦 Instalación y Requisitos

Para ejecutar los códigos, asegúrate de descomprimir `Dataset.zip` en la raíz del proyecto.

Se recomienda tener instalado:
* Python 3.x
* Librerías principales: `pandas`, `numpy`, `sklearn`, `tensorflow/torch` (según corresponda para BERT), `gensim` (para Word2Vec/Doc2Vec).
* R (para los scripts de estadística).

---

## 📝 Cita y Contacto

Si utilizas este código o los resultados para tu investigación, por favor cita el artículo correspondiente o contacta al autor a través de este repositorio.
