🌍 Tourism Experience Analytics

A data analytics and machine learning project that analyzes tourism
behavior, predicts visitor travel mode, and recommends attractions based
on historical preferences.

📌 Project Overview

Tourism Experience Analytics combines Data Cleaning, EDA, SQL,
Machine Learning, and Streamlit to turn tourism transaction data into
useful insights and personalized recommendations.

The project uses user, attraction, location, visit-mode, rating, and
transaction data.

🎯 Objectives

Understand visitor behavior and tourism patterns

Analyze ratings, popular attractions, months, years, and seasons

Predict the visitor's Visit Mode

Recommend attractions using historical preferences

Build an interactive Streamlit dashboard

🛠️ What I Did

1. Data Loading

Loaded multiple Excel tables containing transactions, users,
attractions, cities, countries, regions, continents, visit modes, and
attraction types.

2. Data Cleaning

Checked and handled: - Missing values - Duplicate records - Data types -
Inconsistent attraction categories - Table relationships - Rating and
date-related values

3. Data Integration

Joined the separate tables using their ID relationships and created a
master tourism dataset containing 52,930 transactions.

4. Exploratory Data Analysis

Analyzed: - Rating distribution - Visit Mode distribution - Most popular
attractions - Average ratings - Yearly and monthly visits - Seasonal
tourism patterns

5. Feature Engineering

Created features such as: - Season - Years Since Start - User
Transaction Count - Attraction Popularity - Attraction Average Rating

6. Classification

Built models to predict five Visit Modes: Business, Couples, Family,
Friends, and Solo.

I compared Logistic Regression and Random Forest. The Random
Forest model achieved approximately 40.10% test accuracy.

7. Recommendation System

Created a personalized content-based recommendation system using: -
TF-IDF - Cosine Similarity - Historical user ratings - Average
attraction rating - Attraction popularity

The system generates a ranked list of attractions for a user.

8. Streamlit Dashboard

Created an interactive dashboard where users can: - Enter a User ID -
Select visit year and month - Select an attraction - Predict Visit
Mode - Get personalized recommendations - Explore tourism charts and
insights

🧠 Project Workflow

Raw Excel Data
      ↓
Data Cleaning
      ↓
Data Integration
      ↓
EDA & Visualization
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Prediction + Recommendation
      ↓
Streamlit Dashboard

📊 Key Results

Metric                                 Result

Total Transactions                     52,930
Total Users                            33,530
Attractions in source table             1,698
Attractions in transactions                30
Visit Modes                                 5
Rating Scale                             1--5
Classification Model            Random Forest
Test Accuracy                        ~40.10%
Recommendation Method           Content-Based

The 1,698 figure is the number of attractions in the source attraction
table; 30 distinct attractions are represented in the transaction
records.

📁 Project Structure

Tourism_Analytics_Project/
├── data/
├── database/
├── models/
├── notebooks/
├── app.py
├── requirements.txt
├── tourism_analysis.ipynb
└── README.md

💻 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

TF-IDF

Cosine Similarity

Streamlit

SQL

Joblib

Jupyter Notebook

🚀 How to Run

# Clone the repository
git clone https://github.com/yuvrajgameti09/Tourism-Experience-Analytics.git

# Enter project folder
cd Tourism-Experience-Analytics

# Create virtual environment
python -m venv venv

# Activate environment on Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit application
python -m streamlit run app.py

📈 Future Improvements

Improve Visit Mode classification performance

Handle class imbalance with additional techniques

Build a hybrid recommendation system

Add collaborative filtering

Add maps and attraction images

Add more interactive visualizations

Deploy the dashboard online

Use user feedback to improve recommendations

✅ Conclusion

This project demonstrates an end-to-end tourism analytics and machine
learning workflow. I converted multiple raw Excel tables into a
connected dataset, cleaned and analyzed the data, created useful
features, developed a Visit Mode classification model, and built a
personalized attraction recommendation system.

Finally, I integrated the results into a Streamlit dashboard so
users can interact with predictions, recommendations, and tourism
insights.
