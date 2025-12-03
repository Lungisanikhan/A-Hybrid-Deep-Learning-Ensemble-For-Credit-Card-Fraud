# Enhancing Semantic Relatedness for Low-Resource African Languages

## Overview
This repository contains the implementation and results for the project **Enhancing Semantic Relatedness for Low‑Resource African Languages via Transfer Learning and M2M‑100 Data Augmentation**. The study investigates methods to improve Semantic Textual Relatedness (STR) models for low‑resource African languages.

## Key Features
- 🌍 Focus on multiple African languages
- 🔁 Transfer Learning using multilingual models
- 📈 Data augmentation via Meta AI’s M2M‑100
- 🧪 Full experimental pipeline included

## Repository Structure
```
├── data/                 # Raw and augmented datasets  
├── notebooks/            # Jupyter notebooks for training & evaluation  
├── models/               # Saved model checkpoints  
├── results/              # Evaluation outputs and graphs  
└── src/                  # Core training and preprocessing scripts
```

## Methodology
1. Baseline STR model training on original datasets  
2. Data augmentation using M2M‑100 multilingual translation  
3. Transfer learning with pretrained multilingual models  
4. Performance comparison before and after augmentation  

## Results Summary
- Significant STR performance improvements across evaluated languages  
- Demonstrated the effectiveness of multilingual augmentation  
- Findings support transfer learning as a key strategy for low‑resource NLP  

## How to Use
### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Run Preprocessing
```
python src/preprocess.py
```

### 3. Train Models
```
python src/train.py
```

### 4. Evaluate
```
python src/evaluate.py
```

## Citation
If you use this work, please cite the project accordingly.

## Contributors
- Lungisani Khanyile  
- Research partners & collaborators
