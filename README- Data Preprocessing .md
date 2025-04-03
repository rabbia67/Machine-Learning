# Data Preprocessing for Machine Learning 
This repository contains the code and explanation for a data preprocessing pipeline as part of a Machine Learning course assignment. The preprocessing steps help prepare raw data for training models by cleaning, transforming, and normalizing the dataset to improve model performance.
# 📌 Objectives
- Load and explore the dataset.
- Handle missing data.
- Encode categorical features.
- Normalize numerical values.
- Split the dataset for training and testing.

# 🛠️ Preprocessing Steps
# 1. Import Libraries
Essential libraries such as pandas, numpy, sklearn, and matplotlib are used for data manipulation, preprocessing, and visualization.
# 2. Load the Dataset
``` import pandas as pd
``` data = pd.read_csv("your_dataset.csv")

# 3. Exploratory Data Analysis (EDA)
- Check for null values and data types.
- Summary statistics using .describe() and .info().
- Visualizations like histograms and boxplots for distribution and outliers.

# 5. Categorical Data Encoding
Use:
- LabelEncoder for ordinal data
- OneHotEncoder for nominal data

# 6. Feature Scaling
Normalize features using:
- StandardScaler (Z-score normalization)
- MinMaxScaler (scaling between 0 and 1)

# 7. Train-Test Split
Split the data using train_test_split from sklearn.model_selection.

# ✅ Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn

Install dependencies using:
pip install -r requirements.txt

# 📚 References
- Scikit-learn documentation: https://scikit-learn.org/
- Pandas documentation: https://pandas.pydata.org/

# 📩 Contact
For questions or suggestions, feel free to open an issue or reach out via email.
-Email: raabi.waheed@gmail.com
