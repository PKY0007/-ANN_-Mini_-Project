# -ANN_-Mini_-Project
A machine learning project that predicts loan approval or rejection using an Artificial Neural Network(ANN).The model uses applicant details such as loan amount, CIBIL score, loan term, and assets to make prediction and evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.

Objective:

The main objective is to build a simple binary classification model that can-

* Predict loan approval or rejection.
* Identify important features affecting loan decisions.
* Train an ANN using applicant data.
* Evaluate the model using multiple classification metrics.
  
Project Workflow

Step 1: Dataset Collection

The loan approval dataset is provided as a CSV file containing applicant information and the corresponding loan status. The dataset contains 4,269 loan applications.

Step 2: Data Cleaning

The dataset is checked for duplicate records and unnecessary identifier information. The Loan_ID column is not used as a prediction feature.

Step 3: Data Preprocessing

The input data is prepared for the ANN. Categorical variables are converted into numerical values, missing values are handled, and numerical features are standardized.

Step 4: Training and Testing Split

The dataset is randomly divided into:

80% training data

20% testing data

The training data is used to teach the ANN, while the testing data is used to evaluate its performance.

Step 5: Feature Selection

The relationship between each input feature and the loan approval target is calculated. The most relevant features are selected while avoiding highly correlated duplicate information.

Step 6: ANN Model Initialization

The neural network is initialized with an input layer, a hidden layer containing 4 neurons, and an output layer containing one neuron.

Step 7: Forward Propagation

The selected input features are passed through the neural network. The sigmoid activation function is used to calculate the hidden-layer and output-layer values.

Step 8: Error Calculation

The predicted output is compared with the actual loan status using binary cross-entropy loss. This determines how far the model's prediction is from the correct answer.

Step 9: Backward Propagation

The error is propagated backward through the network. The gradients are calculated to determine how the weights and biases should be changed.

Step 10: Weight and Bias Update

The weights and biases are updated using the selected learning rate. This process is repeated for 600 epochs so that the ANN gradually improves its predictions.

Step 11: Loan Approval Prediction

After training, the ANN predicts the loan status for the test data. An output of 0.5 or greater is classified as Approved (Y), while an output below 0.5 is classified as Rejected (N).

Step 12: Performance Evaluation

The final model is evaluated using:

Accuracy Precision Recall Specificity F1 Score Confusion Matrix These measures help determine how effectively the ANN predicts loan approval and rejection.

Step 13: Final Result

The final result consists of the predicted loan status and the performance of the ANN model. The results can be used to understand how effectively the selected applicant features help in predicting loan approval.
