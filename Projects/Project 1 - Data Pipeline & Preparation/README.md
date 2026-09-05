# Data Preparation and Pipelining System

This project is Automated data preparation pipeline built in Python to reads raw data, detects and fixes anomalies, transforms features, and exports a modeling-ready CSV, wrapped into a single callable pipeline function.

## Objectives

- Dataset Understanding
- Identify Anomaly Data, including: `missing values`, `duplicate records`, `unconsistent format data`, etc.
- Perform Data Cleansing and Data Transform
- Perform Pipelining 
> Read Data -> Identify Anomaly -> Cleansing Data -> Transform Data -> Export into CSV Format
- Documenting all process

## Dataset
Dataset that were used coming from [this](https://archive-beta.ics.uci.edu/dataset/10/automobile)!

This data originally came from **UC Irvine Machine Learning Repository**, here's the information from the source about data:

This data is from 1985 Ward's Automotive Yearbook

Consists of three types of entities: 
- The specification of an auto in terms of various characteristics 
- Its assigned insurance risk rating 
- Its normalized losses in use as compared to other cars.  

The second rating corresponds to the degree to which the auto is more risky than its price indicates. Cars are initially assigned a risk factor symbol associated with its price.   Then, if it is more risky (or less), this symbol is adjusted by moving it up (or down) the scale.  Actuarians call this process `symboling`.  A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.

The third factor is the relative average loss payment per insured vehicle year.  This value is normalized for all autos within a particular size classification (two-door small, station wagons, sports/speciality, etc...), and represents the average loss per car per year.

In this project, the file renamed into `automobileEDA_dirty_training`

## Pipeline Architecture
![asd](/Projects/Project%201%20-%20Data%20Pipeline%20&%20Preparation/documentation/data-flow-diagram.png)
## Stage Breakdown
### Library Used
- Pandas: For general data manipulation
- Numpy: For Numeric (Only) data manipulation
- Matplotlib & Seaborn: For show visual graph and chart using python
- Sklearn: Mainly for data modeling especially Machine Learning, but for this project only use for Data Transforming (MinMax Scalling)
-----
### Read Data
This is where all pipelining begin. In this section, data start to read by pandas with simple code:
```python
def read_data(data):
    df = pd.read_csv(data)
    return df
```
Library Used: `Pandas`

-----

### Anomaly Detection
Exploratory Data Analysis (EDA) is a method where you can identify what kind of data it is. 

1. Show the data itself to check the columns, records, format for each data 
```python
df.head()
```
![raw head data](/Projects/Project%201%20-%20Data%20Pipeline%20&%20Preparation/documentation/raw%20head%20data.png)

2. Check data type for each Feature and also notice missing values

```python
df.info()
```
<p float="left">
  <img src="./documentation/raw info data half 1.png" width=350 />
  <img src="./documentation/raw info data half 2.png" width=400 />
</p>

3. Identify the missing values in a Features and Duplicate Records if exist

```python
missing_count = df.isnull().sum()
missing_count[missing_count > 0].sort_values(ascending=False)
duplicate_count = df.duplicated().sum()
```
4. Reformat the value of all categorical value so they have consistent case of words (Lower), but first I need to grouped by categorical and numerical data

```python
numerical_features = df.select_dtypes(include=[np.number]).columns
categorical_features = df.select_dtypes(include=[object]).columns
for cat in categorical_features.columns:
    df_raw[cat] = df_raw[cat].astype(str).str.lower().str.strip()
```
5. Remove Features that in this project not needed
```python
df_raw.drop(columns=['gas', 'transaction_date'])
```
`transaction_date` currently not needed so I drop this for now, `gas` is not needed because the value is overlaped with `diesel` value. When `gas` is 1, the `diesel` is 0 and vice versa

6. Replace one of `engine-type` value, because it can give wrong assumption if the value is not clear what it tell
```python
df_raw['engine-type'].replace({'l': 'inline'})
```

<table>
  <tr>
    <td align="center"><b>Before</b></td>
    <td></td>
    <td align="center"><b>After</b></td>
  </tr>
  <tr valign="middle">
    <td><img src="./documentation/before replace value.png" width="150" /></td>
    <td align="center" style="font-size: 24px; padding: 0 10px;">➔</td>
    <td><img src="./documentation/after replace value.png" width="150" /></td>
  </tr>
</table>

> Final Summary of Data

![summary dataset](./documentation/summary%20dataset%201.png)
----

### Cleansing Data
In data cleansing, I clean **duplicate data** first then **missing values**, in summary above it shows 4 data are detected to be duplicated, here's the data
![duplicate](./documentation/duplicate%20data.png)
After clean the duplicate data, I divide the data into different group based on data type (numerical & categorical). So, I can deal with missing values based on their data type **BUT** turns out that missing values in categorical data are actually exist as a duplicate value before, meaning there are no missing values for categorical anymore

Now for numerical, there are only 4 missing data: 

![missing data](./documentation/missing%20numerical%20data.png)

There are some methods for numerical missing values you can use, since the missing values is less than 1%, I reccomend to use standard method to refill with **median/mean**. To know which one is use, I check the distribution first for these Features

![chart missing values](./documentation/missing%20chart.png)

`Horsepower` and `Price` are *Extremly* skewed and `Stroke` was *Moderate* skewed, but still since 3 of them are skewed, **Median** is the best choice.
```python
for num in df_raw.select_dtypes(include=[np.number]).columns:
    df_raw[num] = df_raw[num].fillna(df_raw[num].median())
missing_numerical_count = df_raw.select_dtypes(include=[np.number]).isnull().sum()
missing_numerical_count[missing_numerical_count > 0].sort_values(ascending=False) 
```

> Final Summary on Clean Data

![clean data](./documentation/cleansing%20method.png)
----
### Transformation Data
Data can only be processed with a value where a computer/machine is understand, mostly in **numbers** which also include *Boolean*, *Decimal/Float* and bigger *Integer* Number. String or Object type of data cannot be processed because Models are based on **mathematical**-calculation.

Dataset used in this project most of them already in *INT* format:
![data type](./documentation/data%20type.png)


#### Categorical Data
So first, I transform all categorical data into numerical but with different method. This can be done with Encoding.

```
Encoding is the process of converting non-numerical labels, text, or category names into numerical values so that machine learning algorithms and statistical models can process them
```
There are few different method od Encode categorical data, based on the distribution and the uniqueness of value, and I use some of them. In this project, I use 3 Method of encoding: 
- `One Hot Encoding`: every unique labels will be created into new column
- `Label Encoding`: every unique labels will transformed into desired value we choose (into numerical data), basically used when the data is ordinal (have meaningful order)
- `Frequency Encoding`: all unique values turn into a decimal value based on how often the values is appeared

I use each method based on how many **unique** value exist **OR** if the value already have labels with orders.

![categorical data](./documentation/value%20counts%20categorical%20data.png)

Based on categorical data aboce, I grouped the method into

```python
frequency_encoding= ['make', 'fuel-system', 'engine-type', 'body-style']
label_encoding= ['num-of-cylinders', 'symboling', 'horsepower-binned']
oh_encoding= ['drive-wheels', 'num-of-doors', 'aspiration', 'engine-location', 'diesel']
```

```python
#One Hot Encoding
df_raw = pd.get_dummies(df_raw, columns=oh_encoding, drop_first=True)

#Frequency Encoding 
for col in freq_encoding:
    freq = df_raw[col].value_counts(normalize=True)
    df_raw[col] = df_raw[col].map(freq)
    
#Mapping for Label Encoding
cylinder_map = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5, 
    'six': 6, 'eight': 8, 'twelve': 12}

#Label Encoding
horsepower_map = {'low': 0, 'medium': 1, 'high': 2}
```

Here's the final data type after transform categorical data
<table>
  <tr>
    <td align="center"><b>Before</b></td>
    <td></td>
    <td align="center"><b>After</b></td>
  </tr>
  <tr valign="middle">
    <td><img src="./documentation/raw info data - Copy.png" width="300" /></td>
    <td align="center" style="font-size: 24px; padding: 0 10px;">➔</td>
    <td><img src="./documentation/final info data.png" width="300" /></td>
  </tr>
</table>

#### Numerical Data
Numerical transform is needed for give fair scale into the model, so model can calculate and compare `price` that has range 1000 - 100.000 with `num-of-doors` which only have value 2 and 4. Here I use MinMax scaler from **scikit-learn**, because:

```
MinMax Scaler bring all features onto a common scale (typically between 0 and 1) so that variables with larger absolute values do not disproportionately dominate machine learning algorithms
```
```python
scaler = MinMaxScaler()
df_raw[new_num_feat] = scaler.fit_transform(df_raw[new_num_feat])
```

> Final Transform of Data

<div style="text-align: center; margin-bottom: 20px;">
  <b>Before</b><br>
  <img src="./documentation/raw head data.png" />
</div>

<div style="text-align: center;">
  <b>After</b><br>
  <img src="./documentation/final head data.png"/>
</div>

### Save Data
after all task is performed, just call the last dataframe and save it into new File CSV

```python
def save_data(data_raw):
    output_path = os.path.join(base_dir, '..', 'Datasets', 'processed', 'automobileEDA_processed.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_raw.to_csv(output_path, index= False)
```

## Pipelining System
This is the final part where all task above are assembled into 1 action function. This is where *Extract, Transform, Load* (ETL) is used when it becomes 1 pipeline system.

- `Extract`: Read data from csv file into dataframe that pandas can handle with python programming
- `Transform`: Identify Anomaly, Cleansing, and perform Encoding and Scalling data (Transform data) with pandas and scikit-learn
- `Load`: Export the final dataframe into csv file again

```python
def pipeline_data(data):
    df = read_data(data)
    df_detection = anomaly_detection(df)
    df_raw = cleansing_data(df_detection)
    df_final = transformation_data(df_raw)
    result = save_data(df_final)

pipeline_data(file_path) #This is the trigger to run the pipeline above
```