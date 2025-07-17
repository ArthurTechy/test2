# AI-Powered Diabetes Risk Prediction System

&#x20;&#x20;

## Overview

A clinically motivated machine learning system designed to predict diabetes risk using clinical and lifestyle indicators. This project uses advanced ensemble learning, medical-cost-aware optimization, and interpretable modeling techniques to support early diabetes screening aligned with current medical standards.

> ⚠️ **Clinical Disclaimer:** This tool is designed for research and clinical decision support. It is not intended to replace a physician's diagnosis. Follow-up diagnostic testing is required.

## 🎯 Key Achievements

- **70.78%** Overall Accuracy (at optimal clinical threshold)
- **85.19%** Sensitivity (Recall) – Prioritizes diabetes detection
- **77.81%** AUC-ROC Score – Strong class separation
- **2.72%** Generalization Gap – Robust model consistency
- **10-Model Ensemble** – Equal-weighted, clinically optimized

## 📊 Dataset Information

| Attribute           | Details                                              |
| ------------------- | ---------------------------------------------------- |
| **Source**          | Pima Indians Diabetes Dataset                        |
| **Sample Size**     | 768 patient records                                  |
| **Features**        | 8 clinical variables + medically engineered features |
| **Target Variable** | Binary diabetes diagnosis (5-year onset)             |
| **Population**      | Pima Indian women (ages 21+)                         |

### Clinical Features

- Pregnancies count
- Glucose concentration
- Blood pressure (diastolic)
- Skin thickness (triceps skinfold)
- Insulin level
- BMI
- Diabetes pedigree function
- Age

> 📚 *Medical feature engineering was guided by clinical literature including ADA diagnostic thresholds and risk factors.* See: ADA. *Diagnosis and Classification of Diabetes Mellitus*, Diabetes Care, 2010. [https://doi.org/10.2337/dc10-S062](https://doi.org/10.2337/dc10-S062)

## 🔬 Methodology

### Machine Learning Pipeline

```
Data Preprocessing → Medical Feature Engineering → Feature Selection → Model Training → Threshold Optimization → Ensemble Learning → Evaluation
```

### Techniques Used (13 Advanced Components):

- Medically informed feature engineering
- Hybrid feature selection (RFE + univariate)
- Cost-sensitive learning (clinical scoring penalties)
- Hyperparameter tuning (expanded grid & random search)
- Threshold optimization (multi-objective with clinical constraints)
- Custom medical scoring function
- Multiple ensemble strategies
- Stratified 5-fold cross-validation
- Learning curve diagnostics
- 17-model portfolio
- Model stability and performance ranking
- Clinically interpretable metric reporting
- Interactive assessment system

> 📚 *Missing values were handled using Multiple Imputation by Chained Equations (MICE).* See: Azur, M.J., et al. (2011). [https://doi.org/10.1002/mpr.329](https://doi.org/10.1002/mpr.329)

## 🤖 Final Model Summary

### ✅ Best Model: **Equal Weight Ensemble**

- **Threshold:** 0.45 (selected via clinical trade-off analysis)
- **Best Custom Score:** 0.5864 (clinically weighted)

### 📊 Performance at Threshold 0.45

| Metric      | Value  |
| ----------- | ------ |
| Accuracy    | 0.7078 |
| F1-Score    | 0.6715 |
| Precision   | 0.5542 |
| Sensitivity | 0.8519 |
| Specificity | 0.6300 |
| AUC-ROC     | 0.7781 |

### 🏥 Clinical Results:

- ✅ 46 diabetics correctly identified
- ✅ 63 healthy patients correctly identified
- ❌ 8 missed diabetic cases (FN)
- ❌ 37 false alarms (FP)

> 📌 *This configuration prioritizes patient safety by reducing false negatives, while balancing false positives to maintain clinical practicality.*

## 📈 Ensemble Model Composition

10 models included:

1. xgb\_conservative
2. ada\_boost\_conservative
3. rf\_balanced
4. xgb\_balanced
5. catboost\_balanced
6. knn\_optimized
7. bagging
8. svm\_calibrated\_conservative
9. gb\_ultra\_conservative
10. lr\_balanced

## 🧠 Feature Insights

- **Top Feature:** SkinThickness
- **Top 3 Features' Contribution:** 66.3% of total importance

## ⚖️ Medical Trade-Offs

- **False Positive Rate:** 0.3700 – Acceptable for screening
- **False Negative Rate:** 0.1481 – Low enough for safety

### 🩺 Based on clinical literature:

- FN is penalized heavier due to risks like diabetic complications (retinopathy, cardiovascular disease).
- FP is tolerated for safety, aligned with AMA and ADA screening principles: *"When in doubt, test further rather than risk a missed diagnosis."*

**References:**

- American Diabetes Association. *Standards of Medical Care in Diabetes—2024.* Diabetes Care, 47(Supplement\_1), S1–S2. [https://doi.org/10.2337/dc24-S001](https://doi.org/10.2337/dc24-S001)
- American Medical Association. *AI in Health Care: Benefits and Risks*. [https://www.ama-assn.org/](https://www.ama-assn.org/)
- Azur, M.J., et al. (2011). *Multiple imputation by chained equations: what is it and how does it work?* Int J Methods Psychiatr Res. [https://doi.org/10.1002/mpr.329](https://doi.org/10.1002/mpr.329)
- ADA. (2010). *Diagnosis and Classification of Diabetes Mellitus.* Diabetes Care. [https://doi.org/10.2337/dc10-S062](https://doi.org/10.2337/dc10-S062)
- Elkan, C. (2001). *The foundations of cost-sensitive learning.* IJCAI. [https://www.ijcai.org/Proceedings/2001/Papers/111.pdf](https://www.ijcai.org/Proceedings/2001/Papers/111.pdf)
- Saito, T., & Rehmsmeier, M. (2015). *Precision-Recall Plot vs ROC Plot for Imbalanced Data.* PLOS ONE. [https://doi.org/10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)

## ✅ Validation Summary

| Model                 | Generalization Gap |
| --------------------- | ------------------ |
| Equal Weight Ensemble | 0.0272             |
| xgb\_conservative     | 0.0127             |

## 🔬 Recommendations & Considerations

- ⚠️ **Do not use this model for diagnosis.** Use it to support early detection.
- ✅ Embed into workflows as a screening triage tool.
- ✅ Pair with confirmatory lab testing protocols.
- 🔁 Retrain with population-specific data before clinical rollout.
- 📋 Maintain clinical oversight in use.

## 📚 Citing This System

> If using for academic, clinical, or regulatory work, cite the original dataset and this model card. Include attribution to the American Diabetes Association and relevant scoring references.

---

\*Updated: June 2025 – by \*[*Chiezie Arthur Ezenwaegbu*](mailto\:chiezie.arthur@gmail.com)

**License:** MIT | **Disclaimer:** For educational and research use only.

