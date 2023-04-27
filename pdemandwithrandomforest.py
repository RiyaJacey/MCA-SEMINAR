
import mysql.connector
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from seaborn import heatmap
from seaborn import countplot

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Connect to the database
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    db='e_store',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# Load the dataset into a pandas DataFrame
query = "SELECT c.`count`,c.`tamount`, c.`pname`, c.`pbrand`, c.`status`,p.`added_on` FROM `tbl_checkout` as c join tbl_payment as p ON p.`pay_id` = c.`pay_id`"
train_data = pd.read_sql(query, con=connection)

# train_data = pd.read_csv('demand1.csv')
print(train_data.info())
print("Shape",train_data.shape)
print("Description:\n",train_data.describe())
print("Null Values:\n",train_data.isnull().sum())
print(train_data['pname'].value_counts())
cor1=train_data[['count','tamount']].corr()
print(cor1)
fig1= plt.figure()

heatmap(cor1,annot=True)
fig2 = plt.figure()

countplot(x='pbrand', data=train_data)
fig4 = plt.figure()

counts = train_data['pname'].value_counts()

# Create a pie chart
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
plt.axis('equal')
plt.show()



label_encoder = LabelEncoder()
train_data['pname'] = label_encoder.fit_transform(train_data['pname'])
train_data['pbrand'] = label_encoder.fit_transform(train_data['pbrand'])

X_train, X_test, y_train, y_test = train_test_split(train_data[['pname', 'pbrand', 'count']], train_data['status'],
                                                    test_size=0.2, random_state=42)
import matplotlib.pyplot as plt

# Plot X_train
fig3 = plt.figure()


plt.scatter(X_train['pname'], X_train['count'], color='blue', label='Train')

# Plot X_test
plt.scatter(X_test['pname'], X_test['count'], color='red', label='Test')

# Add legend and labels
plt.legend()
plt.xlabel('Product Name')
plt.ylabel('Count')

# Show the plot
print(plt.show())



# Plot
# import seaborn as sns
#
# sns.pairplot(train_data, x_vars=['pname', 'pbrand', 'count'], y_vars='status')

# Create and fit a logistic regression model


rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Create and fit the XGBoost Regressor model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)



model = DecisionTreeRegressor()

# Fit the model to the training data
model.fit(X_train, y_train)

# Connect to the database
cnx = mysql.connector.connect(user='root', password='', host='localhost', database='e_store')

# Create a cursor
cursor = cnx.cursor()

# Execute a SELECT statement to retrieve temperature and accelerometer data
# while True:
# time.sleep(3)
query = 'SELECT status FROM tbl_checkout'
cursor.execute(query)
rows = cursor.fetchall()
print("Status:")

for row in rows:
    if row[0] == 0:
        query1 = 'SELECT pname, pbrand, count FROM tbl_checkout'
        cursor.execute(query1)
        rows = cursor.fetchall()

        test_data = pd.DataFrame(rows, columns=['pname', 'pbrand', 'count'])
        test_data['pname'] = label_encoder.fit_transform(test_data['pname'])
        test_data['pbrand'] = label_encoder.fit_transform(test_data['pbrand'])

        # Calculate demand sums for each product
        query1 = 'SELECT pname,count FROM tbl_checkout'
        cursor.execute(query1)
        rows1 = cursor.fetchall()

        demand_sums = {}
        for row1 in rows1:
            if row1[0] in demand_sums:
                demand_sums[row1[0]] += row1[1]
            else:
                demand_sums[row1[0]] = row1[1]

        # Sort the demand_sums dictionary by demand value in descending order and limit the output to top 10 products
        sorted_demand_sums = sorted(demand_sums.items(), key=lambda x: x[1], reverse=True)[:4]
        top_products = [item[0] for item in sorted_demand_sums]

        # Update the status for the top 10 products




        from sklearn.impute import SimpleImputer

        # Define the imputer
        imputer = SimpleImputer(strategy='mean')

        for product in top_products:
            query='select pname,pbrand,count from tbl_checkout'
            cursor.execute(query)
            rows=cursor.fetchall()
            # Create a new DataFrame for each product
            test_data = pd.DataFrame(rows, columns=['pname', 'pbrand', 'count'])

            # product_data = test_data[test_data['pname'] == product][['pname', 'pbrand', 'count']]
            if len(test_data) > 0:  # Check if there are samples available for the current product
                test_data['status'] = test_data['pname'].apply(lambda x: 1 if x == product else 0)

                # Prepare the test data and predict the labels
                test_data['pname'] = label_encoder.fit_transform(test_data['pname'])
                test_data['pbrand'] = label_encoder.fit_transform(test_data['pbrand'])

                X_test = test_data[['pname', 'pbrand', 'count']]
                y_test = test_data['status']
                model.fit(X_test, y_test)
                rf_y_pred = rf_model.predict(X_test)
                xgb_y_pred = xgb_model.predict(X_test)
                y_pred = model.predict(X_test)
                cnx.commit()

                # Calculate the accuracy of the model on the test data
                rf_mse = mean_squared_error(y_test, rf_y_pred)
                xgb_mse = mean_squared_error(y_test, xgb_y_pred)
                model_mse = mean_squared_error(y_test, y_pred)

                accuracy = accuracy_score(y_test, y_pred)


                print('Accuracy for decision tree regressor:', accuracy)
                print('Random Forest Predictions:', rf_y_pred)
                print('XGBoost Predictions:', xgb_y_pred)

                print('Decision Tree Predictions:', y_pred)
                print('Random Forest MSE:', rf_mse)
                print('XGBoost MSE:', xgb_mse)
                print("Decision Tree MSE:", mean_squared_error(y_test, y_pred))
                # Calculate the RMSE for each model
                rf_rmse = np.sqrt(rf_mse)
                xgb_rmse = np.sqrt(xgb_mse)
                model_rmse = np.sqrt(model_mse)

                print("Random Forest RMSE:", rf_rmse)
                print("XGBoost RMSE:", xgb_rmse)
                print("Decision Tree Regressor Model RMSE:", model_rmse)

                # Calculate the R2 score for each model
                rf_r2 = r2_score(y_test, rf_y_pred)
                xgb_r2 = r2_score(y_test, xgb_y_pred)
                model_r2 = r2_score(y_test, y_pred)

                print("Random Forest R2 Score:", rf_r2)
                print("XGBoost R2 Score:", xgb_r2)
                print("Model R2 Score:", model_r2)


                # Calculate the mean absolute error for each model
                rf_mae = mean_absolute_error(y_test, rf_y_pred)
                xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
                model_mae = mean_absolute_error(y_test, y_pred)
                print("Random Forest MAE:", rf_mae)
                print("XGBoost MAE:", xgb_mae)
                print("Model MAE:", model_mae)

                print('Updated Demand Prediction')
                print(test_data)

                # Print the top 10 products with the highest demand
                print("Top 10 products with the highest demand:")
                for product in top_products:
                    print(product)
                for prediction in y_pred:
                    if prediction ==1:
                        print("high demand")
                        update_query = 'UPDATE tbl_checkout SET status = 0'
                        cursor.execute(update_query)
                        update_query1 = 'UPDATE tbl_checkout SET status = 4 WHERE pname IN ({})'.format(
                            ', '.join(['%s'] * len(top_products)))
                        cursor.execute(update_query1, top_products)

                    if prediction ==0:
                        print("Normal Sales")
                        update_query2 = 'UPDATE tbl_checkout SET status = 0 WHERE pname NOT IN ({})'.format(
                            ', '.join(['%s'] * len(top_products)))
                        cursor.execute(update_query2, top_products)

                    # plt.scatter(y_test, y_pred)
                    # plt.plot(np.linspace(0, np.max(y_pred), 100), np.linspace(0, np.max(y_pred), 100), 'r--')
                    # plt.xlabel('Actual')
                    # plt.ylabel('Predicted')
                    # plt.title('Predicted vs Actual Values')
                    # plt.show()


        cnx.commit()

# # Close the cursor and connection

cursor.close()

cnx.close()
