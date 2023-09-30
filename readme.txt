Name: Bongwoo Jeon
Student ID: 01874516

This project is a simple neural network model to predict whether certain mushrooms are poisonous or edible. The dataset used is Mushroom Dataset in the UCI Machine Learning Repository.

Execution
Execute data_split.py to create(split) datasets(.txt file) for training, testing and validation of the neural network. These files are consist of one-hot binary code(0 or 1).
I put 80% of the data in the training set, 10% in the validation set, and 10% in the test set. 

Execute proj_test.py to run the neural network. Output files are training_err.txt, testing_err.txt and val_err.txt. In all these text files, there are the error accuracies of test, training, anv validation.