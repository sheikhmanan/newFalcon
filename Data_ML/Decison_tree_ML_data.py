import subprocess

# Install pandas
subprocess.run(['pip', 'install', 'pandas'])

# Install matplotlib
subprocess.run(['pip', 'install', 'matplotlib'])

# Install numpy
subprocess.run(['pip', 'install', 'numpy'])

# Install scikit-learn
subprocess.run(['pip', 'install', 'scikit-learn'])

# Import the installed packages
import pandas
import matplotlib
import numpy
import sklearn
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from pyodide.http import pyfetch

 

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
await download(path,"drug200.csv")
path="drug200.csv"
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]