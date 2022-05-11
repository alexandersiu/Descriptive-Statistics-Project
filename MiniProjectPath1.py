#IMPORTS:
from statistics import mode
from re import U
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, RidgeCV, LinearRegression
from scipy import stats


'''
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
#===========================================================================
# Gets the data from the csv:
#===========================================================================
def getData():
    dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
    dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
    dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))
    dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))
    #print(dataset_1.to_string()) #This line will print out your data
    return dataset_1

"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""
#___________________________________________________________________________
#Our Program(and code) starts here:
#Questions to keep in mind while coding this project:

"""
1. What are the variables in your dataset? What do they mean (describe the variables that you plan to use)?
2. After reading the questions for the data set you have chosen to work with, provide a summary statistics
   table of the variables you will use. If you need to transform a variable (e.g., Precipitation into a Raining or not raining variable), this variable must be included in the table. You can use any appropriate summary statistics (e.g., mean, standard deviation, mode).
3. Provide a histogram and explain the resulting plot for at least one variable in your dataset
Data provided is:
    Data is given from Friday Apr 1st 2016 - Monday Oct 31st 2016:
    1. High temp and lowest temp per day in degree F
    2. Precipitation: Rain drop in height in inches
    3. Brooklyn Bridge: Bike Usage
    4. Manhattan Bridge: Bike Usage
    5. Williamsburg Bridge: Bike Usage
    6. Queensboro Bridge: Bike Usage
    7. Total: Total Bike Usage
Question 1:
Q1: You want to install sensors on the bridges to estimate overall traffic across all the bridges.
    But you only have enough budget to install sensors on three of the four bridges.
    Which bridges should you install the sensors on to get the best prediction of overall traffic?
Plan for Q1:
    We need to decide which 3 bridges need sensors; we need to decide criteria to judge the 4 bridges, possible criteria is as follows:
    1. Amount of bike usage per bridge(which bridges have the most traffic will make the sensors most useful)
    2. Argument could be made that there would be less bike usage on hotter days or would not allow the sensor to be used to their fullest
    3. Additional argument off of point 2. the sensor may become faulty, require maintence might be something to consider
    4.
"""
# Descriptive Statistics:
def Descript(precip):


    plt.figure(1)
    plt.hist(precip, color = 'green')
    plt.title("Precipitation Frequency from April 1st - October 31st")
    plt.xlabel("Precipitation (Rain drop height in inches)")
    plt.ylabel("Frequency(Number of occurances)")
    plt.savefig("Precip_Histogram.png")
    plt.show()


# Code for Problem 1; Which bridges should you install the sensors on to get the best prediction of overall traffic?:

def CalcMean(brook_count, man_count, will_count, queens_count, total_bike):

    # Find mean of bike usage across all bridges and the total:
    brook_mean = np.mean(brook_count)
    man_mean = np.mean(man_count)
    will_mean = np.mean(will_count)
    queens_mean = np.mean(queens_count)
    total_mean = np.mean(total_bike)

    #Create a histogram of the data:
    bridges = ('Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro')
    x_axis = np.arange(len(bridges))
    avg_bridge = [brook_mean, man_mean, will_mean, queens_mean]

    plt.bar(x_axis, avg_bridge, align ='center', alpha = 1, color = 'green')
    plt.title('Average Bike Usage at Each Bridge')
    plt.xticks(x_axis, bridges)
    plt.xlabel('Bridge Names')
    plt.ylabel('Total Average Bike Usage')


    plt.show()


    #Print out the top 3 bridges based on mean usage:
    mean_list = [brook_mean, man_mean, will_mean, queens_mean] #List of means
    sorted_mean = mean_list #sorts in ascending order
    sorted_mean.sort()

    #print(sorted_mean)

    for j in range(0,4):
        if(sorted_mean[3] == mean_list[j]):
            if(j == 0):
                print("The Brooklyn Bridge, should be funded for sensors\n")
            elif(j == 1):
                print("The Manhattan Bridge, should be funded for sensors\n")
            elif(j == 2):
                print("The Williamsburg Bridge, should be funded for sensors\n")
            elif(j == 3):
                print("The Queensboro Bridge, should be funded for sensors\n")

        elif(sorted_mean[2] == mean_list[j]):
            if(j == 0):
                print("The Brooklyn Bridge, should be funded for sensors\n")
            elif(j == 1):
                print("The Manhattan Bridge, should be funded for sensors\n")
            elif(j == 2):
                print("The Williamsburg Bridge, should be funded for sensors\n")
            elif(j == 3):
                print("The Queensboro Bridge, should be funded for sensors\n")

        elif(sorted_mean[1] == mean_list[j]):
            if(j == 0):
                print("The Brooklyn Bridge, should be funded for sensors\n")
            elif(j == 1):
                print("The Manhattan Bridge, should be funded for sensors\n")
            elif(j == 2):
                print("The Williamsburg Bridge, should be funded for sensors\n")
            elif(j == 3):
                print("The Queensboro Bridge, should be funded for sensors\n")
    pass

#This runs a ridge regression with the given data set and uses 20% as test data, incremneting as it goes, k-fold cross validation,
def trainModel(x_var, y_var, reg_lambda = np.logspace(-2, 3, 20)):
    lin_model = RidgeCV(alphas = reg_lambda, fit_intercept = True, cv = 5) #Performs Cross
    lin_model.fit (x_var,y_var)
    return lin_model

#Finds the square error and the value of r square
def squareErr(x_1, y_1, lin_model):
    y_model = lin_model.predict (x_1)
    r_square = r2_score(y_1, y_model)
    mean_sq_err = mean_squared_error(y_1, y_model)
    return mean_sq_err, r_square

#Linear Regression:
def lin_regress(feature_matrix, y_var, reg_lambda = np.logspace(-2, 3, 20), Print = True):
    std = preprocessing.StandardScaler() #Initialize pre-processor

    norm_feature_matrix = std.fit_transform(feature_matrix)
    norm_y = std.fit_transform(y_var)

    #Test and Train models:
    train_x, test_x, train_y, test_y = train_test_split(norm_feature_matrix, norm_y, test_size = 0.2, shuffle=False)
    lin_model = trainModel(train_x, train_y, reg_lambda)
    mse, r_2 = squareErr(test_x, test_y, lin_model) # MSE and the r suared value
    best_lambda = lin_model.alpha_ #Find the best lmabda value possible
    y_model = lin_model.predict(test_x)
    pol_coeffs = [i for data_ in lin_model.coef_ for i in data_] #Polynomial coeffcients
    y_intercept = lin_model.intercept_ #y_var intercept
    if Print: print ("The best lambda found is:",best_lambda) # Print the best lambda found

    return y_model, test_y, norm_y, r_2, pol_coeffs, y_intercept[0]


#===========================================================================
#Test our code using this line:
#===========================================================================
if __name__ == '__main__':
    #Get data for the precipitation, and temperatures:
    temp_high = list(getData()["High Temp"])
    temp_l = list(getData()["Low Temp"])
    precip = list(getData()["Precipitation"])

    # Get the bike counts for each bridge and the total of the 4 bridges:
    brook_count = list(getData()["Brooklyn Bridge"]) #Brooklyn Bridge Bike Counts
    man_count = list(getData()["Manhattan Bridge"]) #Manhattan Bridge Bike Counts
    will_count = list(getData()["Williamsburg Bridge"]) #Williamsburg Bridge Bike Counts
    queens_count = list(getData()["Queensboro Bridge"]) #Queensboro Bridge Bike Counts
    total_bike = list(getData()["Total"]) #Total Bike Count

    #Descriptive statistics:
    #For the mean of bridges:
    print("Descriptive Statistics Portion:\n")
    print("The average highest temp is:",np.mean(temp_high))
    print("The average lowest temp is:",np.mean(temp_l))
    print("The average precipitation is:",np.mean(precip))
    print(np.mean(brook_count))
    print(np.mean(man_count))
    print(np.mean(will_count))
    print(np.mean(queens_count))
    print(np.mean(total_bike))

    #For the standard deviation:
    print("\nThe Standard deviation:")
    print("The standard deviation highest temp is:",np.std(temp_high))
    print("The standard deviation lowest temp is:",np.std(temp_l))
    print("The standard deviation precipitation is:",np.std(precip))
    print(np.std(brook_count))
    print(np.std(man_count))
    print(np.std(will_count))
    print(np.std(queens_count))
    print(np.std(total_bike))

    """
    For the mode:
    print("\nThe Mode of the data:")
    print("The mode highest temp is:",mode(temp_high))
    print("The mode lowest temp is:",mode(temp_l))
    print("The mode precipitation is:",mode(precip))
    print(mode(brook_count))
    print(mode(man_count))
    print(mode(will_count))
    print(mode(queens_count))
    print(mode(total_bike))
    """

    plt.hist(precip, color = 'green')
    plt.title("Precipitation Frequency from April 1st - October 31st")
    plt.xlabel("Precipitation (Rain drop height in inches)")
    plt.ylabel("Frequency(Number of occurance)")
    plt.show()


    #Problem 1:
    #We will test them based on which bridges most reliably have the greatest bike usage:
    #We will take the first 3 bridges with the greatest avg mean as they are reliable to have the most bikers ensuring the sensors are fully utilized:
    print("\n\nProblem 1 Portion:")
    CalcMean(brook_count, man_count, will_count, queens_count, total_bike)


    #Problem 2:
    print("\n\nProblem 2 Portion:")

    #Used reshape
    dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))

    #Use np.array
    size = len(dataset_1);
    temp_h = np.array (dataset_1["High Temp"]).reshape(size, 1)
    temp_l = np.array (dataset_1["Low Temp"]).reshape(size, 1)
    precipitation = np.array (dataset_1["Precipitation"]).reshape(size, 1)
    bike_tot = np.array (dataset_1["Total"]).reshape(size, 1)

    #Find the best tested lambda:
    feature_matrix = np.concatenate([temp_h, temp_l, precipitation], axis=1)
    reg_lambda = np.logspace(-2, 3, 20)

    #Find the linear lin_regress with
    y_model, test_y, norm_tot_bikers, r_2, pol_coeffs, y_intercept = lin_regress(feature_matrix, bike_tot, reg_lambda)

    print ("\nModel co-relating the bikers and the precipitation is as follows:\n")
    print("The Coefficient of the respective polynomials are: ")
    print (str(pol_coeffs[0]) + " High Temp + " + str(pol_coeffs[1]) + " Low Temp " + str(pol_coeffs[2]) + " Precipitation "+ str(y_intercept))


    print ("Coefficient of determination of the model is:",r_2) # Coefficient of determination

    plt.plot(test_y, color = "black", label = "Actual Data")
    plt.plot(y_model, 'green', label = "Amount of Bike Users Predicted")
    plt.title("Actual Data vs. Model for Amount of Bike Users Predicted")
    plt.xlabel("Day")
    plt.ylabel("Amoun of Bike Users")
    plt.legend(loc = "best")
    plt.show()

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
def getData():
    dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
    dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
    dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))
    dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))
    #print(dataset_1.to_string()) #This line will print out your data
    return dataset_1


"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""

#this function creates a polyfit graph of the independent (x) and dependent (y) variables up to the degrees wanted
def Polyfit(X, Y, degrees):
    param = []
    y1, y2, y3, y4, y5 = [], [], [], [], []
    y_list = [y1, y2, y3, y4, y5]
    
    for j in degrees:
        mat = feature_matrix(X, j)
        param.append(least_squares(mat, Y))
        
    x_sort = sorted(X)

    #creates 5 equations of degrees 1 to 5 of line of best fit
    for l in x_sort:
        r1 = param[0][0] * l + param[0][1]
        r2 = param[1][0] * (l ** 2) + param[1][1] * l + param[1][2]
        r3 = param[2][0] * (l ** 3) + param[2][1] * (l ** 2) + param[2][2] * l + param[2][3]
        r4 = param[3][0] * (l ** 4) + param[3][1] * (l ** 3) + param[3][2] * (l ** 2) + param[3][3] * l + param[3][4]
        r5 = param[4][0] * (l ** 5) + param[4][1] * (l ** 4) + param[4][2] * (l ** 3) + param[4][3] * (l ** 2) + param[4][4] * l + param[4][5]
        all_r = [r1, r2, r3, r4, r5]
        for i in range(5):
            y_list[i].append(all_r[i])
    
    #creates scatter plot with dataset
    plt.scatter(X, Y, color='b', marker='*')
    
    #assigning each polynomial equation with a different color
    color = ['b', 'g', 'y', 'r', 'm']
    reg_list = [y1, y2, y3, y4, y5]
    ind = 0

    #plots each of the degrees equations on the graph
    for x in range(5):
        plt.plot(sorted(X), reg_list[ind], color = color[ind], linestyle = '-.')
        ind += 1
    
    #places legend, and both axis labels
    plt.legend(["Data Points", "Degree = 1", "Degree = 2", "Degree = 3", "Degree = 4", "Degree = 5"], loc='upper right')
    plt.title("Amount of total bikes vs Amount of Precipitation (in)")
    plt.ylabel("Amount of Rain Precipitation (in)")
    plt.xlabel("Amount of total bikes")
    plt.show()

    return param

#calculates the x^d till x^0 and returns the list
def feature_matrix(x, p):
    x_list = []
    ind = 0

    for i in x:
        ch_p = p
        x_list.append([])
        while ch_p >= 0:
            x_list[ind].append(i**ch_p)
            ch_p -= 1  
        ind += 1

    return x_list

#creates a least square solution for two variables
def least_squares(x, y):
    X_array = np.array(x)
    Y_array = np.array(y)

    return (np.linalg.inv(X_array.T @ X_array)) @ (X_array.T @ Y_array)

y_axis = list(getData()["Precipitation"])
x_axis = list(getData()["Total"])
y_flip = [float(m) for m in y_axis]
x_flip = [float(m) for m in x_axis]

#creates degrees list and runs the polyfit function 
degrees = [1, 2, 3, 4, 5]
paramFit = Polyfit(x_flip,y_flip, degrees)

y_axis = list(getData()["Precipitation"])
x_axis = list(getData()["Total"])
y_flip = [[float(m)] for m in y_flip]
x_flip = [[float(m)] for m in x_flip]
#uses the sklearn function 'train_test_split' and 'LinearRegression' to create a prediction regression model

train_x, test_x, train_y, test_y = train_test_split(x_flip, y_flip, random_state = 0)
regress = LinearRegression(fit_intercept = True)
regress.fit(train_x, train_y)

#creates predictive model 
predict_test = regress.predict(test_x)

#outputs the MSE and R^2 values
print(f'MSE: {mean_squared_error(test_y, predict_test):.4f}')
print(f'R squared value: {r2_score(test_y, predict_test):.4f}')

#creates scatter point with model
plt.scatter(train_x, train_y, color='y')
plt.scatter(test_x, test_y, color='g')
plt.plot(test_x, predict_test, color='blue', linewidth=1)
plt.legend(["training data points", "test data points", "model"])
plt.title("Amount of total bikes vs Amount of Precipitation (in)")
plt.ylabel("Amount of Rain Precipitation (in)")
plt.xlabel("Amount of total bikes")
plt.show()