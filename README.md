# Ex.No.1---Data-Preprocessing
##AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


##ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

##PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
```
#read the dataset
df=pd.read_csv('/content/Churn_Modelling (1).csv')
df
```
```
#drop unwanted columns
df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
```
```
#checking for null, duplicates, outliers in lasrt column
df.isnull().sum()
df.duplicated()
df['Exited'].describe()
```
```
#normalising data to normal distribution
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df),columns=['CreditScore','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'])
df2
```
```
#split dataset
x=df2.iloc[:,:-1].values #all rows from all except last column
x
```
```
y=df2.iloc[:,-1].values #all rows from only last column
y
```
```
##creating training and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
```
```
print(X_test)
print("Size of X_test: ",len(X_test))
```


##OUTPUT:
##Dataset and Its Properties
![nn1](https://user-images.githubusercontent.com/94505585/229990709-ed819ea3-8398-4758-88fa-417056732052.jpg)

<img width="467" alt="image" src="https://user-images.githubusercontent.com/94505585/229993372-c584b5df-2a6a-443c-86fa-be47d00bab91.png">


![nn3](https://user-images.githubusercontent.com/94505585/229990856-84991772-254f-4f41-9066-6508e0713b31.jpg)

##Normalised Dataset
![nn4](https://user-images.githubusercontent.com/94505585/229991923-4904b51e-0f5f-461f-9243-8b2a5ac8c365.jpg)


##X & Y Column Data
![nn5](https://user-images.githubusercontent.com/94505585/229991849-905f60c6-a881-4891-af34-0df7ffa2c6f2.jpg)

![nn6](https://user-images.githubusercontent.com/94505585/229991210-8cd921c4-935d-41fa-83dd-94005054898e.jpg)

##Training Data 
![nn7](https://user-images.githubusercontent.com/94505585/229991382-cf7d2c31-3399-47b1-9a3b-d75d2a768519.jpg)

##Test Data
![nn8](https://user-images.githubusercontent.com/94505585/229991426-383c045a-cbeb-4959-8976-72e3cb93c1d9.jpg)




##RESULT
Thus, the Data preprocessing is performed over a data set successfully.
