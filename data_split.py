import pandas as pd
import numpy as np

#Reading data file
df = pd.read_csv('agaricus-lepiota.data')

dataset = df.iloc[:,0:]
#One hot encoding
dataset = pd.get_dummies(dataset).astype(float)
dataset.head()

#Split the data into training(80), validation(10) and test(10) datasets
train, validate, test = np.split(dataset.sample(frac=1), [int(.8*len(dataset)), int(.9*len(dataset))]) 

#Convert data to int
train = train.astype(int)
validate = validate.astype(int)
test = test.astype(int)

#Save the datasets
train.to_csv('training.txt', index=False, header=None,) 
validate.to_csv('val.txt', index=False, header=None,) 
test.to_csv('testing.txt', index=False, header=None,)