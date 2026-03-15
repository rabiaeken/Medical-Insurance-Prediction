# 🏥 Medical Insurance Cost Prediction

This project is a comprehensive Machine Learning application developed to analyze and predict annual healthcare insurance costs based on personal and demographic factors.

---

### 1. Which Problem Does It Solve?
Insurance companies and individuals often struggle to estimate future healthcare costs accurately. This project addresses the challenge of **cost uncertainty** by:
* Predicting annual medical expenses based on key variables like BMI, smoking status, and age.
* Identifying the most significant risk drivers that influence insurance premiums.
* Providing a data-driven approach to financial planning in the healthcare sector.

### 2. Which Technologies Were Used?
The project is built using the **Python** ecosystem and industry-standard libraries:
* **Data Manipulation:** `Pandas`, `NumPy`
* **Visualization:** `Seaborn`, `Matplotlib`, `Missingno`
* **Machine Learning:** `Scikit-Learn` (Linear Regression, Random Forest, KNN)
* **Preprocessing:** `RobustScaler`, `OneHotEncoder`, `SimpleImputer`

### 3. How Does the System Work?
The system follows a structured Data Science pipeline:
1. **Exploratory Data Analysis (EDA):** Visualizing correlations between costs and lifestyle habits (e.g., the strong impact of smoking on medical expenses).
2. **Outlier Management:** Implementing the **IQR (Interquartile Range)** method to handle extreme values and ensure model stability.
3. **Feature Engineering:** Creating high-impact metrics such as `healthcare_utilization` and `chronic_disease_count` to capture complex health profiles.
4. **Model Evaluation:** Comparing multiple algorithms using **MAE, RMSE,** and **R²** scores to identify the most robust predictor.

### 4. Is It Scalable?
Yes, the architecture is designed for future enhancements:
* **Cloud Deployment:** The script can be transitioned into a REST API (using FastAPI or Flask) and deployed on AWS or Google Cloud.
* **Algorithm Expansion:** Advanced gradient boosting models like XGBoost or LightGBM can be seamlessly integrated into the current pipeline.
* **Deep Learning:** The dataset can be further analyzed using Neural Networks for more complex pattern recognition.

---

## 👥 Contributors
This project was co-authored and developed by **Rabia Eken** and **Çağla Özen**. 
The entire lifecycle, from data cleaning to model implementation, was a collaborative effort.

---
*Developed as part of a Data Science & Machine Learning study.*
