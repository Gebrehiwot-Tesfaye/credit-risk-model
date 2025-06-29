# INTERIM REPORT

## Synthesis of Business Context

In today's digital financial landscape, accurately assessing credit risk is essential for both regulatory compliance and sustainable business growth. This project is designed to build a transparent, interpretable, and robust credit risk model for a digital financial services platform. Guided by the Basel II Accord, our approach emphasizes not only predictive performance but also the ability to explain and justify risk assessments to stakeholders and regulators. The absence of a direct "default" label in the data has required creative, data-driven solutions to engineer a meaningful proxy for credit risk.

---

## Introduction – Project Overview and Understanding

The objective of this project is to segment customers by their credit risk using transaction data. The dataset contains detailed records of customer transactions but does not include a direct indicator of default or risk. To address this, we have structured our work into clear, modular tasks, ensuring that each step is reproducible, business-relevant, and technically sound.

---

## Methodology

### Task 1: Data Understanding & EDA

- Explored the dataset to understand its structure, data types, and key variables.
- Performed summary statistics, visualized distributions, and checked for missing values and outliers.
- All exploratory logic was implemented in reusable Python functions within `src/data_processing.py` and demonstrated in Jupyter notebooks for clarity and reproducibility.

### Task 2: Feature Engineering

- Developed a robust, automated feature engineering pipeline using `sklearn.pipeline.Pipeline`.
- Created aggregate features such as total, average, count, and standard deviation of transaction amounts per customer.
- Extracted temporal features (hour, day, month, year) from transaction timestamps.
- Encoded categorical variables and handled missing values using imputation strategies.
- Scaled numerical features to ensure model readiness.
- All feature engineering steps are script-based, ensuring automation and reproducibility.

### Task 3: Proxy Target Variable Engineering

- Calculated RFM (Recency, Frequency, Monetary) metrics for each customer to capture engagement and transaction behavior.
- Applied KMeans clustering (with scaling and a fixed random state for reproducibility) to segment customers into three groups based on their RFM profiles.
- Identified the "high-risk" group as the cluster with the lowest frequency and monetary value, and created a binary `is_high_risk` target variable.
- Merged this new target back into the main dataset, making it available for downstream model training.

### Task 4: Integration and Reporting

- Integrated all engineered features and the proxy target into a single, model-ready dataset.
- Ensured that all steps are automated, reproducible, and well-documented in both code and notebooks.
- Maintained a high standard of clarity and professionalism in all reporting and code structure.

---

## Challenges & Solutions

- **Lack of a direct credit risk label:** Addressed by engineering a proxy target using RFM metrics and unsupervised clustering.
- **High-cardinality categorical features:** Enhanced plotting and encoding logic to efficiently handle these variables.
- **Data quality and missing values:** Implemented robust imputation and validation steps to ensure data integrity.
- **Reproducibility:** All pipelines and clustering steps use fixed random states and are fully script-based for consistent results.

---

## Future Plan

- **Task 5: Model Training:** Train and evaluate machine learning models using the engineered features and proxy target variable.
- **Task 6: Model Evaluation and Reporting:** Assess model performance, interpretability, and prepare the final project documentation.
- **Timeline:** All remaining tasks are scheduled for completion by Monday.

---

## Conclusion

Substantial progress has been made in understanding the data, engineering meaningful features, and creating a proxy target for credit risk. The project is on track, with a reproducible pipeline and a clear plan for model training and evaluation. I am confident in the successful completion of the project by the planned deadline.

---

_For further details and code, please refer to the project repository: [credit-risk-model](https://github.com/Gebrehiwot-Tesfaye/credit-risk-model)_
