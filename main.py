from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String,MetaData,Float,Table
from sqlalchemy.ext.declarative import declarative_base
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

engine = create_engine('sqlite:///project.db', echo=True)
Base = declarative_base()
class Tables:
    def __init__(self):
        self.x=0
        self.y=0
    
#Table created with the table meta data
#In this step we are creating the database to store the csv data into the database
meta=MetaData()
train = Table(
    'Train',meta,
    Column('x',Float),
    Column('y1',Float),
    Column('y2',Float),
    Column('y3',Float),
    Column('y4',Float),
)

cursor=engine.connect()
meta.create_all(engine)
train = Table(
    'traintest',meta,
    Column('x',Float),
    Column('y',Float),
    Column('delta_y',Float),
    Column('No. of ideal func',Float),
)

def insert_test_train(df):
  df.to_sql('traintest', engine, if_exists='append', index=False)

meta.create_all(engine)

#Create Ideal Table
col_names = ['x'] + [f"y{i}" for i in range(1, 51)]

ideal = Table('Ideal', meta, *[Column(name, Float) for name in col_names])

meta.create_all(engine)


def insert_datatrain(df):
  df.to_sql('Train', engine, if_exists='append', index=False)

def insert_dataideal(df):
  df.to_sql('Ideal', engine, if_exists='append', index=False)

#### Now going to create the dataframe for the ideal Funciton
df=pd.read_csv('train.csv')


insert_datatrain(df)
df=pd.read_csv("ideal.csv")
insert_dataideal(df)
df=pd.read_csv("train.csv")

def linear_regression(x, y):     
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    B0 = y_mean - (B1*x_mean)
    return (B0, B1, f'y = {B0} + {round(B1, 3)}β')

for each in df.columns:
    if each != "x":
        B0, B1, reg_line = linear_regression(df[each], df.x)
        print('Regression Line: ', reg_line)
        plt.figure(figsize=(12,5))
        plt.scatter(df[each], df.x, s=300, linewidths=1, edgecolor='black')
        plt.title(f"Mean Square error: {each}")
        plt.plot(df[each], B0 + B1*df[each], c = 'r', linewidth=5, alpha=.5, solid_capstyle='round')
        plt.scatter(x=df[each].mean(), y=df.x.mean(), marker='*', s=10**2.5, c='r') # average point
        plt.show()

df=pd.read_csv("ideal.csv")
print(df[each])
for each in df:
  if each != 'x':
    mse = mean_squared_error(df[each], df.x)
    print(f"Least Square Error for {each}: {mse}")
    
'''   
The least Square error in the ideal function is y11, so we will consider this to store in our test-data
So lets start storing it in the database for train-test mapping
The function for the delta we use the X's y4
'''

df_train=pd.read_csv("train.csv")
df_lead=pd.read_csv("ideal.csv")
df_test=pd.read_csv("test.csv")
train_delta=df_train['y4']
ideal=df_lead['y11']
trains=[]
ideals=[]
for i in range(0,100):
  trains.append(train_delta[i])
  ideals.append(ideal[i])

df_test['delta_y']=trains
df_test['No. of ideal func']=ideals
insert_test_train(df_test)
df_test.to_csv('ddd.csv')
#So in this way we resolved our assignment, as there are comments and all the data that is added into the database



