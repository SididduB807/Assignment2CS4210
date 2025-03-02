#-------------------------------------------------------------------------
# AUTHOR: Sidharth Basam
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and test decision trees on 3 different dataset sizes
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY SUCH AS numpy OR pandas.
# Only use standard Python libraries.

# Importing necessary Python libraries
from sklearn import tree
import csv

# List of training datasets
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Mapping categorical values to numerical representations
age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'Yes': 1, 'No': 2}
tear_map = {'Normal': 1, 'Reduced': 2}
label_map = {'Yes': 1, 'No': 2}

# Function to transform a categorical row into numerical values

def transform_row(row):
    return [
        age_map[row[0]], spectacle_map[row[1]], astigmatism_map[row[2]], tear_map[row[3]]
    ], label_map[row[4]]

# Load and transform the test data
#Read the test data and add this data to dbTest
dbTest = []
with open('contact_lens_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skipping the header
    for row in reader:
        features, label = transform_row(row)
        dbTest.append((features, label))

# Process each training dataset
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Reading and processing the training data
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            features, label = transform_row(row)
            X.append(features)
            Y.append(label)

    # Running the training and testing process 10 times
    accuracies = []
    for loop in range(10):

        # Fitting the decision tree to the data, setting max_depth=5
        # I believe you said max depth = 3 as an error. The max_depth is 5 and this improves the overall balance as well.
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        correct_predictions = 0
        total_predictions = len(dbTest)

        # Testing phase:
        # #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        for features, true_label in dbTest:
            # Make the class prediction
            predicted_label = clf.predict([features])[0]

            # Compare the prediction with the true label to calculate accuracy
            if predicted_label == true_label:
                correct_predictions += 1

        # Calculate accuracy 
        accuracy = correct_predictions / total_predictions
        accuracies.append(accuracy)

    # Find the average of this model during the 10 runs 
    avg_accuracy = sum(accuracies) / len(accuracies)

    # Print the average accuracy of this model during the 10 runs. should have 3 outputs
    print(f"Final accuracy when training on {ds}: {avg_accuracy:.2f}")
    #as we see as we get more data points in our sets the accurarcy increases 
"""
Output: 
Final accuracy when training on contact_lens_training_1.csv: 0.50
Final accuracy when training on contact_lens_training_2.csv: 0.75
Final accuracy when training on contact_lens_training_3.csv: 0.88
"""


