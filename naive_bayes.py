
#-------------------------------------------------------------------------
# AUTHOR: Sidharth Basam
# FILENAME: naive_bayes.py
# SPECIFICATION: Train and test a Naïve Bayes model on weather data. Use information from other parts of problem 5.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/
import csv
from sklearn.naive_bayes import GaussianNB

# Define mappings to convert categorical data into numerical values
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3} # dictionary based
temperature_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Weak': 1, 'Strong': 2}
play_tennis_map = {'Yes': 1, 'No': 2}
 
# Lists to store training data
X = []  #X is Features
Y = []  #Y is Labels

# Reading and transform the training data. use location of weather_traning.csv
with open("C:\\Users\\justs\\Downloads\\Assignment2CS4210Problem5\\weather_training.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)

    next(reader)  # Skip header

    for row in reader:
        X.append([
            outlook_map[row[1]], 
            temperature_map[row[2]], 
            humidity_map[row[3]], 
            wind_map[row[4]]
        ])
        Y.append(play_tennis_map[row[5]])

# Debug: Print class distribution
yes_count = sum(1 for y in Y if y == 1)
no_count = sum(1 for y in Y if y == 2)
print(f"Training Data Distribution: Yes = {yes_count}, No = {no_count}")

# Train the Naïve Bayes model/apply Naive Bayes
clf = GaussianNB()
clf.fit(X, Y)

# Read and transform the test data
test_data = []
test_instances = []

# predicting the naive bayes
with open("C:\\Users\\justs\\Downloads\\Assignment2CS4210Problem5\\weather_test.csv", 'r') as csvfile:  # FIXED
    reader = csv.reader(csvfile)
    header = next(reader)  # Read and store the header
    
    for row in reader:
        test_data.append(row)
        test_instances.append([
            outlook_map[row[1]], 
            temperature_map[row[2]], 
            humidity_map[row[3]], 
            wind_map[row[4]]
        ])

# Print the header of the output table
print(f"{header[0]:<10} {header[1]:<10} {header[2]:<10} {header[3]:<10} {header[4]:<10} {header[5]:<10} Confidence")

# Use the trained model to predict probabilities for each test instance
predictions = clf.predict(test_instances)
probabilities = clf.predict_proba(test_instances)

# Output results where confidence is >= 0.75
count = 0
for i, row in enumerate(test_data):
    yes_prob, no_prob = probabilities[i]  # Get both probabilities
    predicted_class = predictions[i]
    
    confidence = yes_prob if predicted_class == 1 else no_prob

    if confidence >= 0.75:
        predicted_label = 'Yes' if predicted_class == 1 else 'No'
        print(f"{row[0]:<10} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {predicted_label:<10} {confidence:.2f}")
        
        count += 1  # Keep track of how many have been printed
        if count == 10:  # Stop after printing 10 instances
            break

