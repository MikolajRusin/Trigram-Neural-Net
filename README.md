# 🧠Trigram Neural Net Language Model

An implementation of a simple neural network-based language model, focusing on the trigram concept, inspired by the educational projects of Andrej Karpathy.

## Table of Contents

- [Project Overview](#project-overview)
- [Context and Inspiration](#context-and-inspiration)
- [Model Architecture](#model-architecture)
- [Data and Tokenization](#data-and-tokenization)
- [Example Results](#example-results)

---

## 📜Project Overview

This repository contains a neural network model designed to learn the statistical patterns of a given sequence (characters in names). The model is specifically structured around the **trigram** concept, predicting the third element (the output) based on the preceding two elements.

By using a single-layer neural network with a **one-hot encoded input** instead of a traditional lookup table of counts, the model aims to:

1.  Capture a representation of the input context.
2.  Generalize the principles of the count-based n-gram model using a fully connected layer (where the weights `W` are the parameters being learned).

The final output is a probability distribution over the entire vocabulary, allowing for the generation of new sequences (names).

---

## 💡Context and Inspiration

This project directly addresses a core challenge in Language Modeling: moving from **count-based n-gram models** to **dense, parameter-based neural network models**.

The implementation is heavily inspired by the educational materials and mini-projects developed by **Andrej Karpathy** (e.g., the Makemore series). The goal was to conceptually transition from the traditional count-based Trigram probability matrix (represented by the `N` tensor in the code) to a simple, single-layer neural network (represented by the weight matrix `W`).

### Key Learning Objectives:

* **Tokenization:** Handling character-level tokenization and creating the vocabulary/mapping (`character to integer`, `integer to character`).
* **Trigram Counts:** Building and visualizing the foundational count-based probability distribution.
* **Neural Network as a Lookup:** Implementing a network where the input is a **one-hot encoded context** and the weight matrix `W` directly serves as the learned "lookup table" for log-counts/logits.
* **Training Loop:** Utilizing PyTorch for the forward pass, calculating Cross-Entropy Loss (Negative Log Likelihood), and performing weight updates via gradient descent.

---

## ⚙️Model Architecture

The model is an extremely simple, single-layer neural network that serves a replacement for the traditional count-based Trigram matrix.

### 1. Input Context (X)

* **Size:** The input consists of 702 classes.
* **Encoding:** Each valid two-character context ($C_1C_2$) is converted into a **one-hot encoded vector** of size 702.
> Note: An alternative approach would be to use separate embedding layers for $C_1$ and $C_2$, concatenate their vectors, and feed them into a hidden layer.
However, this implementation uses a single one-hot vector for the combined context $C_1C_2$.
This design ensures that the model **retains positional information**—a context like 'az' is fundamentally different from 'za' and is assigned a unique input index.
Using two separate, concatenated one-hot vectors for $C_1$ and $C_2$ (each of size 27) would result in a combined vector of size 54,
but the simple dot product with the weight matrix $W$ would lose the critical positional distinction needed for a direct lookup.

### 2. The Weight Matrix (W)

The core of the model is a single weight matrix $W$.
* **Dimensions:** $W$ has a shape of **(702, 27)**.
    * **702** rows correspond to the valid input contexts ($C_1C_2$).
    * **27** columns correspond to the output vocabulary (the potential next character $C_3$).

### 3. Output and Loss Function

* **Forward Pass:**
    1.  **Logits:** $L = X_{\text{enc}} \cdot W$
    2.  **Probabilities:** The logits $L$ are passed through the **Softmax** function to convert them into a probability distribution $\hat{P}$ over the 27 possible next characters.
* **Loss Function:** The network is trained using the **Cross-Entropy Loss**, which is equivalent to the **Negative Log Likelihood (NLL)**.
    $$
    \text{Loss} = \text{NLL} + \lambda \sum W^2
    $$
    * A weight decay term ($\lambda = 0.001$) is added to the loss as a simple form of regularization to prevent overfitting.

---
 
## 💾Data and Tokenization

### Data Source
The model was trained on a dataset of approximately **32k** English names, sourced from `names.txt`.

### Character-Level Tokenization
The project uses **character-level tokenization**.
* **Vocabulary Size (V):** 27 unique tokens (26 English letters + the special token `.` representing the start or end of a name).
* **Mapping:** Dictionaries `ch2i` and `i2ch` are used to map characters to integers and vice versa.

### Trigram Context Preparation

The input $X$ for the model is a two-character context ($C_1, C_2$), and the label $Y$ is the next character ($C_3$).

#### Context Reduction (Filtering Impossible Sequences)
In a character-level trigram model using `.` for start/end, any context where the second character $C_2$ is the end token (`.`) is impossible. Since we use `.char` and `char.` as valid boundaries, sequences like `a.` or `m.` (where `.` is $C_2$) should never occur.

To optimize the model input, the original $V \times V = 729$ possible contexts were filtered. All $27$ contexts of the form `$C_1$.` were removed, resulting in a total of **702 valid context sequences** used for training.

---

## ✨Example Results

After training the model for **600** iterations, the network successfully learned the statistical patterns and common trigrams present in the input dataset. 
Below is a chart describing the training and validation losses:
![Training results](images/losses.png)

Following the initial training, the model went through a final **tuning phase** where the **Training** and **Validation** datasets were concatenated (`X_conc_enc, y_conc`). 
The model was trained for an additional **200** iterations on this combined dataset to maximize parameter optimization.
  
The final **Test Loss (NLL)** achieved was approximately **2.35**.

Below are 10 examples of names generated by sampling from the final probability distribution $\hat{P}$ (the weights `W`) of the model:
```text
.akhir.
.vanshri.
.na.
.vallee.
.kayceoran.
.anvea.
.wila.
.emriya.
.kai.
.deeigh.
```
