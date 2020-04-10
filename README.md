# Bank_Customer_Prediction
The goal was to predict how likely a bank's customer will stay or leave in the upcoming days.
An Artificial Neural Network (ANN) is implemented topredict whether a customer will stay with the bank or would leave soon.
Libraries used are: Keras Tensor-flow, scikit-learn, pandas, and numpy.

Three models were implemented-
1. Simple Artificial Neural Network with stochastic gradient descent.
2. ANN using stochastic gradient descent with K-Fold cross validation.
3. Fine Tuning ANN with variations in batch size, epochs, and optimizers.

Results:
Accuracy obtained was the best when the model was fine tuned using GridSearchCv- 85% accuracy
