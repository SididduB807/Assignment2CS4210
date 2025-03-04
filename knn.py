
#-------------------------------------------------------------------------
# AUTHOR: Sidharth Basam
# FILENAME: knn.py
# SPECIFICATION: Implementing 1NN with Leave-One-Out Cross-Validation on email classification csv file
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1
#-----------------------------------------------------------*/
# Importing libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

# Reading the email classification data from CSV file
emailData = []
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Skip the header row
    for row in reader:
        emailData.append(row)

#error counter
errorCount = 0
totalInstances = len(emailData)

# Loop through each instance to perform LOO-CV
for loop in range(totalInstances):

    X_train = []
    Y_train = []
    for j in range(totalInstances):
        if j != loop:  # Exclude the test instance
            features = list(map(float, emailData[j][:-1]))  # Convert features to float
            label = 1 if emailData[j][-1] == "spam" else 0  # Convert labels ("ham" -> 0, "spam" -> 1)
            X_train.append(features)
            Y_train.append(label)

    X_test = [list(map(float, emailData[loop][:-1]))]  # Convert to float
    Y_test = 1 if emailData[loop][-1] == "spam" else 0  # Convert label

    # Train 1NN classifier
    clf = KNeighborsClassifier(n_neighbors=1, p=2)  # p=2 for Euclidean distance
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)[0]

    if Y_pred != Y_test:
        errorCount += 1

# Compute and print the LOO-CV error rate
errorRate = errorCount / totalInstances
print(f"LOO-CV Error Rate for 1NN: {errorRate:.4f}")




