# Data preprocessing phase---->
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Name", OneHotEncoder(), [1])],  remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Data Preprocessing ends----->>>



# Let's make ANN

# import libraries
import keras
from keras.models import Sequential # Used for initializing our NN
from keras.layers import Dense # Used for building hidden layers
from keras.layers import Dropout

classifier = Sequential()

# Initialize the input nodes and 1st hidden layer of the ANN
#output_dim is inupt nodes + output nodes divided by 2 i.e. avg of all the nodes; init is the weights where we specified uniform i.e near to zero; 
#input_dim is the num of input nodes that ae needed to be assigned just to the 1st hidden layer bcz from 2nd hidden layer its input will be the output of 1st hidden layer.
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# Dropout in keras is used to tackle Overfitting problem. It will shut down a few nodes based on value of p in thehidden layers so that our model dosen't get over trained
classifier.add(Dropout(p = 0.1)) 
# add 2nd hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))
# add the output layer
# output_dim will be 1 bcoz our output nodes are binary; activation func for output be sigmoid
# but if we have more calasses than just 2 in our output the use activation = 'softmax' which is also a sigmoid function for more classes.
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile our model
# optimizer is stochastic gradient descent algo named as adam; 
# loss is the error function for binaryoutput but if our o/p has multiple classes the use categorical_crossentropy;
# metrices is used to show the accuracy after each batch.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Fit our training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# To build confusion matrix we need values as true or false. Set threshold = 0.5. i.e >0.5 = True otherwise False
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# External homework prediction of a user 
pred_more = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
new_pred = classifier.predict(sc.transform(pred_more))
new_pred = (new_pred > 0.5)
print(new_pred)



#===========================================>>>>> Using K-Fold Cross validations

# Run data preprocessing steps then move ahead ....

# import libraries
from keras.models import Sequential # Used for initializing our NN
from keras.layers import Dense # Used for building hidden layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# build ANN structure
def build_classifier(): 
    clf = Sequential()
    clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    clf.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
    return clf

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()



# Fine tunnning our model=================>>>>>>
# i.e. testing it with multiple parameters such as batch size, epoch, and optimizers.

from keras.models import Sequential # Used for initializing our NN
from keras.layers import Dense # Used for building hidden layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# build ANN structure
def build_classifier(optimizer): 
    clf = Sequential()
    clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    clf.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    clf.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    clf.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy']) 
    return clf

classifier = KerasClassifier(build_fn = build_classifier)

# set different parameters on which we want to test our model's accuracy
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']} # rmsprop is also based on stochastic GD.
   

grid_search = GridSearchCV(classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ # to get the parameters giving highest accuracy
best_accuracy = grid_search.best_score_






