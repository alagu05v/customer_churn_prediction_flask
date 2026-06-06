# Customer Churn Prediction Web Application

## Overview

This project is a Machine Learning-powered web application that predicts customer churn using the Telco Customer Churn dataset. The application is built using Flask and XGBoost and includes SHAP Explainable AI visualizations to help understand the factors influencing predictions.

---

## Features

- Customer churn prediction using Machine Learning
- Flask-based web application
- XGBoost classification model
- SHAP Explainable AI integration
- User-friendly interface
- Data preprocessing and feature engineering
- SQLite database support

---

## Technologies Used

### Programming Language
- Python

### Machine Learning
- XGBoost
- Scikit-learn
- Pandas
- NumPy

### Explainable AI
- SHAP

### Web Development
- Flask
- HTML
- CSS

### Database
- SQLite

---

## Project Structure

```text
Customerchurn_Flask_XAI/
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── instance/
│   └── database.db
│
├── model/
│   ├── xgb_model.pkl
│   └── columns.pkl
│
├── static/
│   ├── style.css
│   ├── shap_plot.png
│   └── churn_report.pdf
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/alagu05v/customer_churn_prediction_flask.git
```

### Navigate to the Project Directory

```bash
cd customer_churn_prediction_flask
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training using XGBoost
6. Model Evaluation
7. SHAP Explainability
8. Flask Deployment

---

## Screenshots

### Home Page

Customerchurn_F

### Prediction Form

Customerchurn_Flask_XAI/Screenshots/form.png

### Prediction Result

_Add screenshot here_

### SHAP Explainability

_Add screenshot here_

---

## Future Enhancements

- Cloud deployment (AWS/Azure)
- User authentication
- REST API integration
- Advanced analytics dashboard
- Real-time predictions

---

## Author

### Alagu Vallilingam

LinkedIn:  
https://www.linkedin.com/in/alagu-vallilingam-b2206b251

GitHub:  
https://github.com/alagu05v

---

## Project Highlights

- End-to-end Machine Learning project
- Explainable AI using SHAP
- Web application deployment using Flask
- Real-world business problem: Customer Churn Prediction

---

## License

This project is created for educational and portfolio purposes.
