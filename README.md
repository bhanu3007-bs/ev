# ev
EV Adoption Forecasting Using Historical Registration Data

Overview

As the global shift toward sustainable transportation accelerates, electric vehicle (EV) adoption continues to grow at a rapid pace. For policymakers, urban planners, and energy providers, anticipating this growth is critical to avoiding infrastructure shortfalls—particularly in the deployment of EV charging stations. A failure to forecast accurately may result in congestion, long charging wait times, and ultimately, hinder the broader transition to clean mobility.

Problem Statement

To support strategic planning and ensure smooth EV integration into the transportation ecosystem, there is a pressing need for data-driven forecasting. Using historical EV registration data from the Washington State Department of Licensing (DOL)—which tracks monthly registrations from January 2017 to February 2024 across all counties—we aim to develop a regression-based forecasting model.

The dataset contains detailed attributes including:

Vehicle type (Battery Electric Vehicles [BEVs], Plug-in Hybrid Electric Vehicles [PHEVs])

County-level geographical information

Total and non-electric vehicle registrations

Vehicle usage classification (Passenger vs. Truck)

Monthly snapshots of EV penetration and overall adoption trends

Objective

Build a robust regression model that can forecast future electric vehicle adoption across Washington counties. This model should:

Predict total EV registrations in the coming months or years

Identify growth patterns based on geography, vehicle type, and usage

Estimate the increasing share of EVs in the total vehicle population

Serve as a decision-making tool for infrastructure planning, particularly for charging station rollout

Why It Matters

Without proactive planning, EV growth could outpace infrastructure readiness, leading to:

User dissatisfaction due to insufficient charging access

Missed climate targets at local or state levels

Increased grid pressure from uncoordinated charging demand

Forecasting EV adoption accurately enables stakeholders to:

Strategically deploy public and private charging stations

Prepare electrical grids for increased load

Design policies and incentives that align with future mobility trends
Dataset Link: https://www.kaggle.com/datasets/sahirmaharajj/electric-vehicle-population-size-2024/data

1. Install required packages
%pip install joblib

2. Import libraries
import joblib import numpy as np import pandas as pd import seaborn as sns import matplotlib.pyplot as plt from sklearn.preprocessing import LabelEncoder from sklearn.ensemble import RandomForestRegressor from sklearn.model_selection import train_test_split, RandomizedSearchCV from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

3. Load the dataset
from google.colab import files uploaded = files.upload()

import io df = pd.read_csv(io.BytesIO(uploaded['3ae033f50fa345051652.csv']))

4. Display top 5 rows
df.head()

5. Shape of the dataset
print("Shape:", df.shape)

6. Data types and memory usage
df.info()

7. Check for missing values
print("Missing values:\n", df.isnull().sum())

8. Outlier detection in 'Percent Electric Vehicles'
Q1 = df['Percent Electric Vehicles'].quantile(0.25) Q3 = df['Percent Electric Vehicles'].quantile(0.75) IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR upper_bound = Q3 + 1.5 * IQR

print('Lower Bound:', lower_bound) print('Upper Bound:', upper_bound)

outliers = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)] print("Number of outliers in 'Percent Electric Vehicles':", outliers.shape[0])

9. Data cleaning
df['Date'] = pd.to_datetime(df['Date'], errors='coerce') df = df[df['Date'].notnull()] df = df[df['Electric Vehicle (EV) Total'].notnull()]

df['County'] = df['County'].fillna('Unknown') df['State'] = df['State'].fillna('Unknown')

print("Missing after fill:") print(df[['County', 'State']].isnull().sum())

df.head()

10. Cap outliers in 'Percent Electric Vehicles'
df['Percent Electric Vehicles'] = np.where( df['Percent Electric Vehicles'] > upper_bound, upper_bound, np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, df['Percent Electric Vehicles']) )

Confirm no remaining outliers
outliers_capped = df[(df['Percent Electric Vehicles'] < lower_bound) | (df['Percent Electric Vehicles'] > upper_bound)] print("Outliers after capping:", outliers_capped.shape[0])

Requirement already satisfied: joblib in /usr/local/lib/python3.11/dist-packages (1.5.1) 3ae033f50fa345051652.csv 3ae033f50fa345051652.csv(text/csv) - 1216895 bytes, last modified: 7/15/2025 - 100% done Saving 3ae033f50fa345051652.csv to 3ae033f50fa345051652.csv Shape: (20819, 10) <class 'pandas.core.frame.DataFrame'> RangeIndex: 20819 entries, 0 to 20818 Data columns (total 10 columns):

Column Non-Null Count Dtype
0 Date 20819 non-null object 1 County 20733 non-null object 2 State 20733 non-null object 3 Vehicle Primary Use 20819 non-null object 4 Battery Electric Vehicles (BEVs) 20819 non-null object 5 Plug-In Hybrid Electric Vehicles (PHEVs) 20819 non-null object 6 Electric Vehicle (EV) Total 20819 non-null object 7 Non-Electric Vehicle Total 20819 non-null object 8 Total Vehicles 20819 non-null object 9 Percent Electric Vehicles 20819 non-null float64 dtypes: float64(1), object(9) memory usage: 1.6+ MB Missing values: Date 0 County 86 State 86 Vehicle Primary Use 0 Battery Electric Vehicles (BEVs) 0 Plug-In Hybrid Electric Vehicles (PHEVs) 0 Electric Vehicle (EV) Total 0 Non-Electric Vehicle Total 0 Total Vehicles 0 Percent Electric Vehicles 0 dtype: int64 Lower Bound: -3.5174999999999996 Upper Bound: 6.9025
