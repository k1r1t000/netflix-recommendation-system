# Netflix Movie Recommendation System

A personalized movie recommendation system built on the Netflix Prize Dataset using collaborative filtering and neural recommendation models. The project compares Generalized Matrix Factorization (GMF) and Deep Neural Collaborative Filtering (Deep NCF) for personalized movie recommendations.

---

## Results

| Model          | Type                                 |       RMSE |     MAP@10 |
| -------------- | ------------------------------------ | ---------: | ---------: |
| Deep NCF (MLP) | Neural Collaborative Filtering       |     0.8123 |     0.0743 |
| **GMF**        | **Generalized Matrix Factorization** | **0.7717** | **0.1673** |

**Best Model:** GMF

---

## Dataset

Netflix Prize Dataset

Original Dataset Statistics:

* 100,480,507 ratings
* 480,189 users
* 17,770 movies

Filtered Dataset Used for Training:

* 45,290,083 ratings
* 54,404 users
* 7,127 movies

Ratings are on a 1–5 star scale.

---

## Project Structure

```text
netflix-recommendation-system/

├── Netflix_Recommendation_System.ipynb
├── recommend.py

├── models/
│   ├── best_GMF_(Matrix_Factorization).pt
│   └── best_Deep_NCF_(MLP).pt

├── data/
│   ├── combined_data_1.txt
│   ├── combined_data_2.txt
│   ├── combined_data_3.txt
│   ├── combined_data_4.txt
│   ├── movie_titles.csv
│   ├── probe.txt
│   └── qualifying.txt

├── requirements.txt
├── .gitignore
└── README.md
```

---

## Repository Modules (Deliverable 2)

This repository fulfills the required architecture components:

*   **Data Processing Pipeline**: Found in Section 1 of `netflix-recommendation-system.ipynb`. Handles chunked Parquet creation and intelligent threshold filtering.
*   **Model Training Pipeline**: Found in Section 2 of `netflix-recommendation-system.ipynb`. Defines PyTorch Datasets, Dataloaders, GMF, and Deep NCF architectures.
*   **Evaluation Scripts**: Found in Section 3 of `netflix-recommendation-system.ipynb`. Implements RMSE and MAP@10 calculations.
*   **Recommendation Generation Module**: Implemented in the standalone `recommend.py` script.
*   **Documentation & Instructions**: Provided in this `README.md` and the accompanying `technical_report.md`.

---

## Features

* Data preprocessing and filtering
* Exploratory Data Analysis (EDA)
* Generalized Matrix Factorization (GMF)
* Deep Neural Collaborative Filtering (NCF)
* RMSE evaluation
* MAP@10 ranking evaluation
* Top-10 personalized movie recommendations
* Cold-start handling
* Pre-trained model inference

---

## Recommendation Example

For a sample user, the GMF model generated recommendations such as:

* The Wire: Season 2
* The Shield: Season 3
* Seinfeld: Season 4
* Ken Burns' Civil War
* Led Zeppelin

The recommendations are generated using learned user and movie embeddings from historical rating behavior.

---

## Model Architecture

### GMF

* User embedding dimension: 128
* Movie embedding dimension: 128
* Dot-product interaction
* User bias and item bias terms
* Global rating mean offset

### Deep NCF

* User embedding dimension: 64
* Movie embedding dimension: 64
* Multi-layer perceptron:

  * 128 → 256 → 128 → 64 → 1
* ReLU activations
* Dropout (0.2)

---

## Evaluation

Metrics used:

* RMSE (Root Mean Squared Error)
* MAP@10 (Mean Average Precision at 10)

GMF achieved the best performance on both prediction accuracy and recommendation quality.

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

### Option 1: Fast Inference (Recommended)
If you simply want to generate recommendations using the pre-trained weights without waiting hours for the models to train, run the inference script. This will instantly load the `.pt` models from the `models/` directory:

```bash
python recommend.py
```

### Option 2: Train from Scratch
If you want to view the exploratory data analysis, data processing, and retrain the models entirely from scratch in your computer's RAM, run the Jupyter Notebook:

```bash
jupyter notebook netflix-recommendation-system.ipynb
```

---

## Acknowledgements

Dataset: Netflix Prize Dataset

Competition:
http://www.netflixprize.com
