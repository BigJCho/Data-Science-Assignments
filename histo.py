from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('iris.csv')
slength= data['sepal_length']
swidth= data['sepal_width']
plength= data['petal_length']
pwidth= data['petal_width']

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.hist(slength, bins = 15, color='#974bd1', edgecolor='#d4b620')
plt.title('Sepal Length')
plt.xlabel('Centimeter')
plt.ylabel('Frequency')
plt.xticks(np.arange(min(slength), max(slength) + 0.5, 0.5))

plt.subplot(2,2,2)
plt.hist(swidth, bins = 15, color='#a94bd1', edgecolor='#d4b620')
plt.title('Sepal Width')
plt.xlabel('Centimeter')
plt.ylabel('Frequency')
plt.xticks(np.arange(min(swidth), max(swidth) + 0.25, 0.25))

plt.subplot(2,2,3)
plt.hist(plength, bins = 15, color='#4bd17a', edgecolor='#d4b620')
plt.title('Petal Length')
plt.xlabel('Centimeter')
plt.ylabel('Frequency')
plt.xticks(np.arange(min(plength), max(plength) + 0.5, 0.5))

plt.subplot(2,2,4)
plt.hist(pwidth, bins = 15, color='#5dd14b', edgecolor='#d4b620')
plt.title('Petal Width')
plt.xlabel('Centimeter')
plt.ylabel('Frequency')
plt.xticks(np.arange(min(pwidth), max(pwidth) + 0.25, 0.25))

plt.tight_layout()
plt.show()
