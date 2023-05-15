# import scipy.io
# mat = scipy.io.loadmat('/Users/varunchowdary/Desktop/Desktop/Sem-8/Cog sci and Ai/project/Brain2Word_paper-master/data/subjects/M04/data_180concepts_pictures.mat')
# print(mat)

#M04


import numpy as np
#train
data_train = np.load('output/M02/1.npy')
data_train = data_train[0]
print(data_train.shape)
print(data_train)
#test
data_test = np.load('output/M02/2.npy')
data_test = data_test[0]
print(data_test.shape)
#train
target_train = np.load('output/M02/3.npy')
target_train = target_train[0]
print(target_train.shape)
#test
target_test = np.load('output/M02/4.npy')
target_test = target_test[0]
print(target_test.shape)


import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

# Encode the target variables using LabelEncoder
le = LabelEncoder()

all_classes = set()

for i in target_train:
    for j in i:
        all_classes.add(j)

le.fit(list(all_classes))

target_train_total = []
target_test_total = []
for i in target_train:
    target_train_encoded = le.transform(i)
    target_train_total.append(target_train_encoded)
for i in target_test:
    target_test_encoded = le.transform(i)
    target_test_total.append(target_test_encoded)

target_test_encoded = target_test_total
target_train_encoded = target_train_total


# Train a multi-output linear regression model on the training data
model = MultiOutputRegressor(LinearRegression())
model.fit(data_train, target_train_encoded)

# Use the trained model to make predictions on the test data
predicted_target_test_encoded = model.predict(data_test)

# Decode the predicted target variables back to their original form
predicted_target_test = le.inverse_transform(predicted_target_test_encoded)

# Evaluate the performance of the model on the test data
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(target_test, predicted_target_test)
print("Mean Squared Error on Test Data:", mse)
