# 🏥 Medical Insurance Cost Prediction

A machine learning web application that predicts annual medical insurance charges based on customer information such as age, BMI, smoking status, gender, number of children, and region.

The project covers the complete machine learning workflow—from exploratory data analysis and feature engineering to model selection, hyperparameter tuning, and deployment using Streamlit.

> **Note:** This model is trained on the **Medical Cost Personal Dataset (US)**. The predicted values represent annual medical insurance charges from this dataset and are displayed in **US Dollars (USD)**.

---

## 🚀 Live Demo

> Add your deployed Streamlit link here after deployment.

```
https://your-streamlit-app.streamlit.app
```

---

## 📷 Application Preview

> Add screenshots after deployment.

| Home | Prediction |
|------|------------|
| ![Home](<img width="1470" height="835" alt="image" src="https://github.com/user-attachments/assets/78b9bda4-89f9-4f0f-ada1-8fecd1004bb3" />
) | ![Prediction](images/prediction.png) |

---

# 📌 Project Overview

This project predicts annual medical insurance charges using supervised machine learning regression algorithms.

The objective is to estimate insurance costs based on an individual's demographic and lifestyle information.

The project includes:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Multiple Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Hyperparameter Tuning using GridSearchCV
- Model Evaluation
- Streamlit Deployment

---

# 📂 Dataset

**Dataset:** Medical Cost Personal Dataset (US)

### Features

- Age
- Sex
- BMI
- Children
- Smoker
- Region

### Target Variable

- Medical Insurance Charges

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

# 📊 Machine Learning Workflow

### Phase 1

- Import Libraries

### Phase 2

- Load Dataset

### Phase 3

- Exploratory Data Analysis (EDA)

### Phase 4

- Data Cleaning

### Phase 5

- Feature Engineering

- One-Hot Encoding using `pd.get_dummies()`

### Phase 6

- Train-Test Split

### Phase 7

Model Training

- Multiple Linear Regression

### Phase 8

Model Evaluation

Metrics used:

- MAE
- MSE
- RMSE
- R² Score
- Adjusted R²

### Phase 9

Variance Inflation Factor (VIF)

### Phase 10

Cross Validation

### Phase 11

Ridge Regression

### Phase 12

Lasso Regression

### Phase 13

ElasticNet Regression

### Phase 14

Hyperparameter Tuning

- GridSearchCV

### Phase 15

Model Comparison

The best-performing model was selected based on evaluation metrics.

✅ Final Model: **Lasso Regression**

### Phase 16

Model Interpretation

### Phase 17

Save Trained Model

Using:

- joblib

Saved Files

```
insurance_lasso_model.pkl

feature_columns.pkl
```

### Phase 18

Streamlit Deployment

- Interactive User Interface
- Real-time Prediction
- Responsive Design

---

# 📁 Project Structure

```
Medical-Insurance-Cost-Prediction/

│

├── app/
│   └── app.py
│
├── data/
│   └── insurance.csv
│
├── images/
│
├── models/
│   ├── insurance_lasso_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── Medical_Insurance_Cost_Prediction.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Medical-Insurance-Cost-Prediction.git
```

Move into the project

```bash
cd Medical-Insurance-Cost-Prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app/app.py
```

---

# 📈 Model Information

| Property | Value |
|----------|--------|
| Problem Type | Regression |
| Final Algorithm | Lasso Regression |
| Encoding | pd.get_dummies() |
| Hyperparameter Tuning | GridSearchCV |
| Deployment | Streamlit |

---

# 🔮 Future Improvements

- Support additional datasets
- Add model comparison dashboard
- Explain predictions using SHAP values
- Deploy using Docker
- Integrate cloud database
- Add user authentication

---

# 📄 License

This project is developed for educational and portfolio purposes.

---

# 👨‍💻 Author

**Rahul Simhadri**

Computer Science Student

Interested in Artificial Intelligence, Machine Learning, and Full-Stack Development.

LinkedIn: *(Add your LinkedIn URL)*

GitHub: *(Add your GitHub URL)*
