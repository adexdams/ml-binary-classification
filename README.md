# ml-binary-classification
The model predicts by classification the likelihood of a customer to make a purchase. 


### **Problem Definition**

This is a binary classification ML model development task. The model should predict with fair accuracy (at least 60%), the likelihood of a customer purchasing a product (target variable) if they were recommended that product.



### **About the Data**

This dataset contains information on customer purchase behavior across various attributes, aiming to help data scientists and analysts understand the factors influencing purchase decisions. The dataset includes demographic information, purchasing habits, and other relevant features.

It was sourced from **Kaggle**: https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset



### **About the Features**
* **Age:** Customer's age
* **Gender:** Customer's gender (0: Male, 1: Female)
* **Annual Income:** Annual income of the customer in dollars
* **Number of Purchases:** Total number of purchases made by the customer
* **Product Category:** Category of the purchased product (0: Electronics, 1: Clothing, 2: Home Goods, 3: Beauty, 4: Sports)
* **Time Spent on Website:** Time spent by the customer on the website in minutes
* **Loyalty Program:** Whether the customer is a member of the loyalty program (0: No, 1: Yes)
* **Discounts Availed:** Number of discounts availed by the customer (range: 0-5)
* **Purchase Status (Target  Variable):** Likelihood of the customer making a purchase (0: No, 1: Yes)

### Approach

1. Started with an exploratory data analysis to understand the data.
2. Sought missing values but I had nothing to do next since there were none.
3. Checked for outlier values using min/max and describe the function.
3. Developed pie charts, box plots, bar charts
4. A correlation table and heat map were developed to see if the variables are correlated and to what extent.
5. Prepared the data and developed 4 classification ML models to assess the best predictive estimator.
6. The estimators developed and the related accuracy scores were Gaussian Naive Bayes (0.81), Linear SVC (0.58), KNeighbors Classifier (0.62), and RandomForestClassifier (0.93).
7. I selected the RandomForestClassifier because it had the highest score of 0.93
8. Took on a hyperparameter tunning of the RandomForestClassifier and achieved a marginal 0.01 improvement of the accuracy score.


### Conclusion

I selected the RandomForestClassifier because it had the highest accuracy score of 0.94. However, I should mention I worry the RandomForestClassifier model might have been exposed to the data previously because it was the only model that I used the cross-validation approach upon while training the data. I might be wrong about the overfitting issue, nevertheless, the RandomForestClassifier was the best pick of all.
