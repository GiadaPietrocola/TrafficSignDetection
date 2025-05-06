# 🚦 Rilevamento di Segnali Stradali con OpenCV e SVM

## Panoramica del Progetto

Questo progetto si concentra sul rilevamento di segnali stradali, in particolare segnali di **“Senso Vietato”**, utilizzando tecniche tradizionali di visione artificiale tramite **OpenCV** e una **Support Vector Machine (SVM)** per la classificazione.  
A differenza degli approcci basati su deep learning, questo metodo punta a maggiore efficienza computazionale e interpretabilità, risultando adatto per applicazioni leggere o embedded.

![img](https://github.com/user-attachments/assets/6247afdf-179b-43af-a218-8200c717f271)

---

## ✨ Caratteristiche Principali

- Pre-elaborazione immagini:
  - Sfocatura Gaussiana
  - Sogliatura adattiva
  - Operazioni morfologiche
- Estrazione delle ROI:
  - Rilevamento di rettangoli e cerchi mediante contorni e Trasformata di Hough
- Segmentazione avanzata:
  - Region Growing su spazio colore HSV
- Estrazione di feature:
  - Descrittori HOG (Histogram of Oriented Gradients)
- Classificazione:
  - SVM binaria (segnali vs. non-segnali)
- Valutazione:
  - Curve Precision-Recall su dataset standard e personalizzati

---

## 📂 Dataset

### Dataset Primario

- Immagini pubbliche di segnali stradali (es. GTSRB, LISA)

### Dataset Personalizzato ("Cassino Dataset")

- 25 immagini reali di segnali “Senso Vietato” catturate a Cassino, Italia
- Condizioni variabili di illuminazione e angolazione

---

## 🔬 Metodologia

### 1. Pre-elaborazione

- Conversione in scala di grigi
- Sfocatura Gaussiana (kernel 5x5, sigma = 0.6)
- Operazione morfologica *Top-hat* (elemento 11x61)
- Sogliatura adattiva per binarizzazione

### 2. Rilevamento dei Rettangoli

- Estrazione contorni e filtraggio geometrico:
  - Area: 800–12.000 px²
  - Rapporto d’aspetto: 0.75–1.6
  - Orientamento: 45°–135°

### 3. Segmentazione HSV (Region Growing)

- Conversione delle ROI in spazio colore HSV
- Segmentazione tramite soglie sul canale hue (rosso) e saturazione
- Dilatazione iterativa (massimo 60 iterazioni)

### 4. Rilevamento dei Cerchi

- Trasformata di Hough:
  - `minDist = 120`
  - Raggio: 20–150 px
- Validazione dei cerchi che racchiudono i rettangoli rilevati

### 5. Estrazione delle Feature e Classificazione

- Ridimensionamento delle ROI a 80x80 px
- Estrazione HOG:
  - Dimensione cella: 8x8
  - Dimensione blocco: 2x2
  - Numero di bin: 9
- Addestramento e test della SVM con cross-validation a 5 fold

---

## 📊 Risultati

- **AUC Precision-Recall**:
  - 0.994 su dataset pubblico
  - 1.000 su dataset personalizzato di Cassino
- **Prestazioni reali**:
  - 100% di rilevamento sui segnali “Senso Vietato” reali

---

## 📌 Riferimenti

- [Documentazione OpenCV: HOG Descriptors](https://docs.opencv.org)
- Forbes, *Statistiche sulla Guida Distratta* (2024)

---

## 👩‍💻 Autori

- **Giada Pietrocola**
- **Achille Cannavale**

**Istituzione**: Università degli Studi di Cassino e del Lazio Meridionale  

---

> 🚗 Un'alternativa leggera, interpretabile ed efficace al deep learning per il rilevamento di segnali stradali!




