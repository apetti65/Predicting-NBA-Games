import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('games.csv')

#g = sns.PairGrid(dataset[['PTS_home', 'PTS_away', 'HOME_TEAM_WINS', 'HOME_ORTG', 'AWAY_ORTG', 'HOME_DRTG', 'AWAY_DRTG', 'HOME_SRS', 'AWAY_SRS', 'HOME_EFG', 'AWAY_EFG', 'HOME_TS', 'AWAY_TS']])
#g = g.map_upper(plt.scatter,marker='+')
#g = g.map_lower(sns.kdeplot, cmap="hot",shade=True)
#g = g.map_diag(sns.kdeplot, shade=True)
#plt.savefig('plots.png')
#plt.show()

neg, pos = np.bincount(dataset['HOME_TEAM_WINS'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

regression_dataset = dataset.iloc[:, 7:]
class_dataset=dataset.iloc[:, 7:-1]

# Use a utility from sklearn to split and shuffle your dataset.
train_cdf, test_cdf = train_test_split(class_dataset, test_size=0.2)

train_clabels = np.array(train_cdf.pop('HOME_TEAM_WINS'))
test_clabels = np.array(test_cdf.pop('HOME_TEAM_WINS'))

train_cfeatures = np.array(train_cdf)
test_cfeatures = np.array(test_cdf)

initial_bias = tf.keras.initializers.Constant(np.log([pos/neg]))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(7, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax', bias_initializer=initial_bias))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.fit(train_cfeatures, train_clabels, epochs=150, batch_size=len(train_cfeatures), verbose=2)
yhat = model.predict(test_cfeatures)
yhat = np.argmax(yhat, axis=-1).astype('int')
acc = accuracy_score(test_clabels, yhat)
print("Accuracy: "+str(acc))
#model.save('classification_model.h5')

train_rdf, test_rdf = train_test_split(regression_dataset, test_size=0.2)

    #Form np arrays of labels and features.
train_rlabels = np.array(train_rdf.pop('MOV'))
test_rlabels = np.array(test_rdf.pop('MOV'))

train_rfeatures = np.array(train_rdf)
test_rfeatures = np.array(test_rdf)

regression_model = tf.keras.models.Sequential()
regression_model.add(tf.keras.layers.Dense(7, activation='relu'))
regression_model.add(tf.keras.layers.Dense(1))
regression_model.compile(loss='mean_absolute_error', optimizer='adam')

regression_model.fit(train_rfeatures, train_rlabels, epochs=125, batch_size=len(train_rfeatures), verbose=2)
reg = regression_model.evaluate(test_rfeatures, test_rlabels, verbose=2)

#regression_model.save('regression_model.h5')