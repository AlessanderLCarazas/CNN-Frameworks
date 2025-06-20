# Clasificación CNN para Datasets de Imágenes Médicas y de Dibujos Animados

Este proyecto utiliza redes neuronales convolucionales (CNN) para clasificar imágenes de tres conjuntos de datos diferentes: SimpsonsMNIST, BreastMNIST y HAM10000. El proyecto está implementado usando PyTorch e incluye scripts para la preparación de datos, entrenamiento de modelos y evaluación.

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Conjuntos de Datos](#conjuntos-de-datos)
- [Scripts](#scripts)
  - [Script de Preparación de Datos](#script-de-preparación-de-datos)
  - [Script Principal Completo](#script-principal-completo)
  - [Script de Ejecución Simple](#script-de-ejecución-simple)
- [Resultados](#resultados)
- [Instrucciones de Uso](#instrucciones-de-uso)
- [Dependencias](#dependencias)
- [Características del Modelo CNN](#características-del-modelo-cnn)
- [Nota sobre HAM10000](#nota-sobre-ham10000)

---

## Descripción del Proyecto

Este proyecto tiene como objetivo resolver problemas de clasificación utilizando redes neuronales convolucionales (CNN) en tres conjuntos de datos diferentes. El objetivo es evaluar el rendimiento de los modelos CNN basado en las siguientes métricas:

- Precisión (Precision)
- Exactitud (Accuracy)
- Sensibilidad (Recall)
- Puntuación F1 (F1-Score)

---

## Conjuntos de Datos

1. **SimpsonsMNIST:** Conjunto de datos de imágenes de los personajes de Los Simpson, categorizados por personaje.
   - Fuente: [SimpsonsMNIST GitHub](https://github.com/alvarobartt/simpsons-mnist)

2. **BreastMNIST:** Conjunto de datos de mamografías para clasificación binaria (Normal vs. Anormal).
   - Fuente: [MedMNIST](https://medmnist.com/)

3. **HAM10000:** Conjunto de datos de imágenes dermatoscópicas de lesiones cutáneas, categorizadas por tipo de lesión.
   - Fuente: [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

---

## Scripts

### Script de Preparación de Datos

**Archivo:** `data_preparation.py`

Primero ejecuta este script para verificar que tus datos estén correctamente estructurados.

Este script prepara los conjuntos de datos para el entrenamiento y la evaluación, crea directorios necesarios y valida la estructura de los datos.

### Script Principal Completo

**Archivo:** `cnn_classification.py`

Contiene la arquitectura completa de la CNN, el manejo de los datasets, el entrenamiento y la evaluación detallada del modelo.

### Script de Ejecución Simple

**Archivo:** `simple_execution_script.py`

Este script automatiza el entrenamiento y la evaluación con configuraciones simplificadas y visualizaciones integradas.

---

## Resultados

### SimpsonsMNIST

![image](https://github.com/user-attachments/assets/503f9f7f-f8cc-4fb6-a97c-2712f282941d)
![image](https://github.com/user-attachments/assets/3ce2c683-3959-4685-abdd-b855d4ba9dd2)


### BreastMNIST

![image](https://github.com/user-attachments/assets/4636587b-de96-4c4d-8677-ec63b933aab9)
![image](https://github.com/user-attachments/assets/49ecea04-0c65-42c5-9318-d828b8d61417)



### FINAL RESULTS SUMMARY
![image](https://github.com/user-attachments/assets/05948aa9-550f-4bf9-9159-4101df532156)
![image](https://github.com/user-attachments/assets/79aaae3c-7dc5-4dcc-b664-d26f22657122)



---

## Lo que se obtendra

### Métricas completas para cada dataset:
- Accuracy
- Precision
- Recall
- F1-Score

### Visualizaciones:
- Matrices de confusión
- Gráfico comparativo de métricas
- Progreso de entrenamiento

### Archivos de resultados:
- `cnn_results.csv` - Resultados finales
- Imágenes de matrices de confusión
- Gráfico comparativo de métricas

---

## Dependencias

- torch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- Pillow
- opencv-python

---

## Características del Modelo CNN

- **Arquitectura:** 4 bloques convolucionales con BatchNorm
- **Regularización:** Dropout para evitar overfitting
- **Optimización:** Adam optimizer con scheduler
- **Data Augmentation:** Rotaciones, flips, color jitter
- **Métricas:** Accuracy, Precision, Recall, F1-Score

