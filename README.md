# Binary-Decision-Tree
Binary Decision Tree Implemented in Python without scikit or Pandas using Numpy version 1.7.1.

Given Problem:

In decisionTree.fpy j java j cpp j mg, implement a Decision Tree learner. This file should learn a decision
tree with a specified maximum depth, print the decision tree in a specified format, predict the labels of the
training and testing examples, and calculate training and testing errors.
Your implementation must satisfy the following requirements:

• Use mutual information to determine which attribute to split on.
• Be sure you’re correctly weighting your calculation of mutual information. For a split on attribute X,
I(Y ,X) = H(Y ) - H(Y | X) = H(Y ) - P(X = 0)H(Y|X = 0) - P(X = 1)H(Y|X = 1).
Equivalently, you can calculate I(Y ,X) = H(Y ) + H(X) - H(Y,X).
• As a stopping rule, only split on an attribute if the mutual information is > 0.
• Use a majority vote of the labels at each leaf to make classification decisions.

• Six command-line arguments are: \<train input\> \<test input\> \<max depth\> \<train out\> \<test out\> \<metrics out\>. These arguments are described
in detail below:
1. \<train input\>: path to the training input .csv file.
2. \<test input\>: path to the test input .csv file.
3. \<max depth\>: maximum depth to which the tree should be built
4. \<train out\>: path of output .labels file to which the predictions on the training data should
be written.
5. \<test out\>: path of output .labels file to which the predictions on the test data should be
written.
6. \<metrics out\>: path of the output .txt file to which metrics such as train and test error should
be written.

handout folder contains some example data sets to work on.
