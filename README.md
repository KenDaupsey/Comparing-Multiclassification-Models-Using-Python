# Multiclass Classification Model Comparison
This repository contains a comprehensive project that evaluates and compares the performance of various multiclass classification models on a given dataset related to economic well-being. The project is implemented using Python and popular machine learning libraries like scikit-learn, pandas, and matplotlib.

## Project Description
The goal of this project is to compare the predictive performance of several multiclass classification models, including Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Support Vector Machine. The models are trained and evaluated on a dataset containing features such as Age, Education Level, Marital Status, Credit Score, Annual Income, Savings Amount, and Income Category. The target variable is the Health Status, which is a multiclass variable.

The project follows a structured approach to preprocess the data, split it into training and testing sets, train the models, and evaluate their performance using metrics such as accuracy, classification report, and confusion matrix. The results are displayed in a tabular format and visualized using a bar plot for easy comparison.

## Dataset
The dataset used in this project is the EconomicWellbeing_Data.csv file, which is available in the repository. It contains information about individual economic well-being factors and their corresponding health status.

To load dataset, do this:
### Load Dataset
import pandas as pd ## If you haven't already done so
url= "https://raw.githubusercontent.com/KenDaupsey/Multiclass-Classification-using-Logistic-Regression/main/EconomicWellbeing_Data.csv"
df= pd.read_csv(url)
df.head()

## Requirements
To run this project, you need to have the following dependencies installed:
### Python 3.x
### pandas
### numpy
### scikit-learn
### matplotlib
### seaborn

You can install the required packages using pip:
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn

## Usage
Clone this repository to your local machine.
Navigate to the project directory.
Open the Jupyter Notebook or Python script file.
Run the cells or script to execute the code.
The code will perform the following steps:

Import the necessary libraries.
Load the dataset.
Select the features and target variable.
Split the dataset into training and testing sets.
Define the preprocessing steps (scaling and one-hot encoding).
Create a dictionary of classifiers.
Loop through each classifier, train the model, and evaluate its performance.
Display the results in a tabular format.
Visualize the accuracy comparison of the models using a bar plot.

### Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
This project was inspired by the need to compare the performance of different multiclass classification models and provide a comprehensive example for educational purposes.
