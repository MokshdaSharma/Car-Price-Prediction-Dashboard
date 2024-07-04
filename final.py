import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import IPython.display as display
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import zscore
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


st.set_page_config(
    page_title="Car Prices Prediction Model",
    page_icon="✅",
    layout="wide",
)

st.title("Car Price Prediction")


df = pd.read_csv("C:\\Users\Mokshda Sharma\Desktop\My Projects\Car_prices\dataset.csv", encoding = "ISO-8859-1")

st.header('Correlation Heatmap')
st.write('''
                The target variable price is highly related to curbweight, enginesize, and horsepower. 
                citympg and highwaympg are highly correlated to each other. 
                carlength and carwidth are highly correlated to each other.
                wheelbase and car length highly correlated to each other.
                Features like carid, symboling shows least correlation ''')

heatmap_data = df.select_dtypes(include=['number']).corr()
# plt.figure(figsize=(13,6))

corr = heatmap_data.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Draw the heatmap
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={"shrink": .8}, ax=ax)

# Set the title
# plt.title('Relationship between food and health', fontsize=16)

# Display the heatmap in Streamlit
# st.title("Correlation Heatmap")
st.pyplot(fig)

# sns.heatmap(heatmap_data, vmin=-1, cmap='coolwarm', annot=True)
# st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# st.title('Basic EDA :chart_with_upwards_trend:')
# st.header('Target/Dependent Variable: Price')
col1, col2 = st.columns((2))
with col1: 
    st.header('Target/Dependent Variable: Price')
    st.write('Skewed form of target variable')

with col2:
    plt.figure(figsize=(8,5))
    plots = sns.histplot(df['price'],color="y")
    plt.xlabel('Price', weight = 'bold')
    plt.ylabel('Count', weight = 'bold')
    # plt.title('', weight = 'bold')
    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.0f'), 
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                   ha='center', va='center', size=10, xytext=(0, 6), 
                   textcoords='offset points')
    st.pyplot()

# st.header('Analysis of Categorical ')
# col1, col2 = st.columns((2))
# df["Order Date"] = pd.to_datetime(df["Order Date"])

numerical_features = ['wheelbase', 'carheight', 'curbweight',
                      'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                      'peakrpm', 'citympg', 'highwaympg']
categorical_features = df.describe(include=['object','category']).columns

# with col1:
#     st.header('features')
#     st.write(' **Description of categorical features** ')
#     st.write(''' fueltype: Car fuel type i.e gas or diesel 

# aspiration: Aspiration used in a car 

# doornumber: Number of doors in a car 

# carbody: body of car

# drivewheel: type of drive wheel 

# enginelocation: Location of car engine 

# enginetype: Type of engine

# cylindernumber: cylinder placed in the car

# fuelsystem: Fuel system of car ''')

# with col2:
#     cat_f = st.selectbox('**Select Categorical Feature for bar plot**', categorical_features)
#     st.write(f'**Relation between {cat_f} and Car Prices**')
#     plt.figure(figsize=(10, 5))
#     plt.bar(df[cat_f], df['price'], alpha=0.7)
#     plt.xlabel(cat_f, weight = 'bold')
#     plt.ylabel('Car Prices', weight = 'bold')
#     plt.title(f'{cat_f} vs Car Prices', weight = 'bold')
#     st.pyplot()
    


# st.header('Analysis of Numerical ')
# col1, col2 = st.columns((2))
# with col1:
#     st.header('features')
#     st.write(' **Description of numerical features** ')
#     st.write(''' wheelbase: Weelbase of car \n
# carlength: Length of car \n
# carwidth: Width of car \n
# carheight: height of car \n
# curbweight: The weight of a car without occupants or baggage \n
# boreratio: Boreratio of car \n
# stroke: Stroke or volume inside the engine \n
# compressionratio: compression ratio of car\n
# horsepower: Horsepower \n
# peakrpm: car peak rpm \n
# citympg: Mileage in city \n
# highwaympg: Mileage on highway \n
# enginesize: Size of car ''')

# with col2:
#     num_f = st.selectbox('**Select Numerical Feature**', numerical_features)
#     st.write(f'**Analysis of distribution of {num_f}**')
#     plt.figure(figsize=(10, 8))
#     sns.histplot(data=df[num_f], bins=20, kde=True, palette='cool')
#     plt.title(f'{num_f}', weight = 'bold')
#     st.pyplot()



# col1, col2 = st.columns((2))
# with col1:
#     avg_prices_by_car = df.groupby('company')['price'].mean().sort_values(ascending=False)

# # Plot top N car models by average price
#     n = 20  # Number of top car models to plot
#     top_car_models = avg_prices_by_car.head(n)
#     st.write(' **Avg price of cars of various brands** ')
#     plt.figure(figsize=(6, 4))
#     sns.barplot(x=top_car_models.values, y=top_car_models.index, palette='Reds')
#     plt.title(f'Top {n} Car Models by Average Price', weight = 'bold')
#     plt.xlabel('Average Price', weight = 'bold')
#     plt.ylabel('Car Model', weight = 'bold')
#     plt.tight_layout()  
#     st.pyplot()

# with col2:
#     n = 20  # Number of top car models to plot
#     top_car_models = df['company'].value_counts().head(n)
#     st.write('**Most purchased car models of various brands**')
#     plt.figure(figsize=(6, 4))
#     sns.barplot(x=top_car_models.values, y=top_car_models.index, palette='Greens')
#     plt.title(f'Top {n} Car Models by Frequency')
#     plt.xlabel('Frequency')
#     plt.ylabel('Car Model')
#     plt.tight_layout()
#     st.pyplot()


df_autox = pd.DataFrame(df.groupby(['company'])['price'].mean().sort_values(ascending = False))
df_autox.rename(columns={'price':'price_mean'},inplace=True)
df = df.merge(df_autox,on = 'company',how = 'left')

df['company_cat'] = df['price_mean'].apply(lambda x : 0 if x < 12000 else (1 if 12000 <= x < 24000 else 2))

dataset_pr = df.copy()

st.title('Outliers :crossed_flags:')
st.write(''' The following columns contains outliers:
            - curbweight
            - peakrpm
            - price
            - price_mean
            - car_area''')
plt.figure(figsize=(20, 5))
#to check the outliers
sns.boxplot(data=dataset_pr)
# Rotate x labels to prevent overlapping
plt.xticks(rotation=30)
st.pyplot()

a = dataset_pr[['curbweight', 'peakrpm', 'price', 'price_mean', 'car_area']].quantile(0.25)
b = dataset_pr[['curbweight', 'peakrpm', 'price', 'price_mean', 'car_area']].quantile(0.75)
IQR = b - a

# Calculate lower and upper bounds
lower_bound = a - 1.5 * IQR
upper_bound = b + 1.5 * IQR

# Create a boolean DataFrame indicating outliers
outliers = (dataset_pr[['curbweight', 'peakrpm', 'price', 'price_mean', 'car_area']] < lower_bound) | (dataset_pr[['curbweight', 'peakrpm', 'price', 'price_mean', 'car_area']] > upper_bound)

# Any row that has a True value in any column is an outlier
outliers = outliers.any(axis=1)

# Filter out the outliers
dataset_pr = dataset_pr[~outliers]

#label encoding
encoders_nums = {"fueltype":{"diesel":1,"gas":0},
                 "aspiration":{"turbo":1,"std":0},
                 "doornumber":     {"four": 4, "two": 2},
                 "enginelocation":     {"front": 1, "rear": 2},
                 "drivewheel":{"fwd":0,"4wd":0,"rwd":1},
                 "cylindernumber":{"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }
                 }

df = dataset_pr.replace(encoders_nums)

# One hot encoding
df = pd.get_dummies(df, columns=["carbody", "enginetype","fuelsystem"], prefix=["body", "etype","fsystem"])

features = numerical_features.copy()
features.extend(['fueltype','aspiration','doornumber','drivewheel','cylindernumber','company_cat','body_convertible',
       'body_hardtop', 'body_hatchback', 'body_sedan', 'body_wagon','etype_dohc', 'etype_l', 'etype_ohc', 'etype_ohcf',
       'etype_ohcv','fsystem_1bbl', 'fsystem_2bbl'
       , 'fsystem_idi', 'fsystem_mpfi',
       'fsystem_spdi'])

X = df[features].apply(zscore)

y = np.log10(df['price'])


# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

carbody_columns = ['body_sedan', 'body_hatchback', 'body_hardtop', 'body_wagon', 'body_convertible']
df['carbody'] = df[carbody_columns].idxmax(axis=1).str.replace('carbody_', '')

# Sidebar filters
st.sidebar.header("Filter Options")

# Filter options for citympg
citympg_min, citympg_max = st.sidebar.slider('City MPG', min_value=int(df['citympg'].min()), max_value=int(df['citympg'].max()), value=(int(df['citympg'].min()), int(df['citympg'].max())))

# Filter options for horsepower
horsepower_min, horsepower_max = st.sidebar.slider('Horsepower', min_value=int(df['horsepower'].min()), max_value=int(df['horsepower'].max()), value=(int(df['horsepower'].min()), int(df['horsepower'].max())))

# Filter options for fueltype
fueltype = st.sidebar.multiselect('Fuel Type', options=df['fueltype'].unique(), default=df['fueltype'].unique())

# Filter options for doorno
doorno = st.sidebar.multiselect('Number of Doors', options=df['doornumber'].unique(), default=df['doornumber'].unique())

# Filter options for carbody
carbody = st.sidebar.multiselect('Car Body', options=df['carbody'].unique(), default=df['carbody'].unique())

# Filter the data based on the selections
filtered_data = df[
    (df['citympg'] >= citympg_min) & (df['citympg'] <= citympg_max) &
    (df['horsepower'] >= horsepower_min) & (df['horsepower'] <= horsepower_max) &
    (df['fueltype'].isin(fueltype)) &
    (df['doornumber'].isin(doorno)) &
    (df['carbody'].isin(carbody))
]

# Predict prices using the trained model on filtered data
filtered_data['predicted_price'] = model.predict(filtered_data[features])

# Create a list of the columns used for filtering
filter_columns = ['citympg', 'horsepower', 'fueltype', 'doornumber', 'carbody', 'predicted_price']

# Main content
st.title("Car Specifications and Prices")

# Display only the filtered columns with the predicted prices
st.write(filtered_data[filter_columns])


# Show the average price of the filtered results
col1 , col2 = st.columns((2))
with col1:
    st.write(" **Avg price of filtered results** ")

with col2: 
    if not filtered_data.empty:
        avg_price = filtered_data['price'].mean()
        st.write(f"Average Price: ${avg_price:,.2f}")
    else:
        st.write("No data matches the filters.")