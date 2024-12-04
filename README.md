# Used Car Price Analysis

## Overview

This project aims to analyze a dataset of used cars to identify the key factors that influence their prices. The original dataset contained 3M vehicles, how we are provided with information on 426K cars, including various attributes such as condition, odometer reading, year, and more. The goal is to develop a predictive model to help a used car dealership understand what consumers value in a used car.

### Data Analysis jupyter notebook
The data analysis notebook can be found at this public github repository:
https://github.com/mgk2014/PCMLAI-cars

## Business Understanding

From a business perspective, the task is to identify key attributes that drive the price of used car prices. This involves leveraging the provided vehicles data set, conducting exploratory data analysis, preparing data, employing machine learning for linear, ridge and lasso regression, selecting appropriate hyper parameters and evaluating the models, and providing a recommendation on which model to go with

## Data Preparation Tasks

Data quality and cleanup steps included:

1. Removing IDs, VINs - since these are text columns and would not have an impact on the price
2. Removing categorical columns with a large number of unique values such as state(51), region (404), model(29549). While some models of vehicles in some states may drive price, including all these values, will make the models evaluated below too complicated to run on laptops.
3. Removing columns with more than 50% missing data (size of vehicle - 70% missing values)
4. Dropping duplicate rows (approx 182K rows were duplicate VIN numbers)
5. Converting data types (cylinder from string to numeric)
6. Addressed outliers with price and odometer. For the purposes of this analysis all vehicles price < 100K and odometer <500k are included

## Findings

Initial data analysis provided the following assessment:
1. Vehicles with `lower odometer` readings, and `newer year models` command higher values. As the odometer readings increase the condition of the vehicles determine the value of the vehicle. Vehicles in `new, like new conditions` command higher prices
2. More `recent year vehicles sell for more` than than older vehicles
3. Vehicles with higher number of cylinders command higher prices. On average vehicles with `6, 8 cylinders sell for 15k and 17.5k` respectively, compared to 4 cylinder that sell for ~11k
3. Vehicles in `new and like new conditions sell on average for ~27k and 19k` respectively. Salvaged vehicles on the other hand sell on average for 3.7k
4. Vehicles with `parts only titles sell for 3.2k` on average

### Machine Learning models

We evaluated  3 different machine learning models (LinearRegression, Lasso, Ridge on both numerical and categorical variables). With LinearRegression, degree 3 model with both numerical and categorical data we achieved a score of 61%. This model may be used to make predictions on the price of a car given the following features: 
1. odometer reading, year, number of cylinders
2. title status, manufacturer, condition, fuel type, vehicle type

As an example, our chosen model predicts the price of 3 vehicles (2005 yr, 6 cyl, clean title, 100k miles, manual, sedan, gas)  as follows:
1. Toyota: 6,389.37
2. Audi: 7,869.89
3. Ford: 5,453.41

We find that the features that have the biggest `positive` impact on the price of the car as follows (from most to least important):

1. Manufacturer: `Ferrari, Tesla, Ashton Martin, Porsche`
2. Fuel Type: `Diesel`
3. Title Status: `Lien`
4. Type: `Pickup`
5. Condition: `like new`

The features that have the biggest `negative` impact on the price of the car as follows (from most to least important):

1. Manufacturer: `Fiat, Harley, Mitsubishi, Kia, Nissan, Chrysler`
2. Fuel Type: `Electric`
3. Type: `Bus, hatchback`
4. Title: `parts only`

# Next steps recommendation

1. The original data set had a lot of `missing data, ranging from 20 to 40%` for these columns
* condition       40.79
* cylinders       41.62
* drive           30.59

These features were inferred using most frequently occuring method. Perhaps KNeighborsClassifier may be used to find values for these categorical variables

2. Instead of computing missing values, when all missing values are dropped, the data shrinks to 15% of it's original size. This translates to 60K records. Running the above models in this data data set may provide a better model, assuming the significanly reduced data set can provide a good representation of the overall data. This requires further analysis

3. Additional model optimization may be accomplished by reviewing the outlier conditions, scaling methods and further adjusting the hyper parameters of the model

## Detailed findings and visual explanations


### Exploratory data analysis

This analysis includes exploration of overall data set, bar coupons and cheap restaurant coupons.

1. Univariate analysis
2. Bi-variate analysis (e.g., how cylinders and prices move together)
3. Correlation analysis
4. Visualizations (e.g., scatter plots, histograms, bar plots)

### Modeling Evaluation

Different regression models are built to predict used car prices. The models include:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression

Each model is evaluated using metrics such as RMSE and R-squared. The models are built using both numerical and categorical features.



The models are evaluated based on their performance metrics. The findings are reviewed to determine whether the earlier phases need revisitation and adjustment or if the information is valuable to the client.

## Deployment

The final step is to deliver the findings to the client. The work is organized as a basic report that details the primary findings and provides recommendations to the used car dealership.





## Conclusion

The project provides insights into the key factors that influence used car prices. The models developed can help the dealership fine-tune their inventory and pricing strategies based on consumer preferences.

## Files

- `notebook.ipynb`: The Jupyter Notebook containing the analysis and modeling
- `data/vehicles.csv`: The dataset used for the analysis
- `images/`: Directory containing images used in the notebook
- `readme.md`: This readme file

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run

1. Clone the repository
2. Install the required packages
3. Open the Jupyter Notebook and run the cells sequentially
