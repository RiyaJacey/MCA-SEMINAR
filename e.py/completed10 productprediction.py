import mysql.connector
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import pymysql
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Connect to the database
# connection = pymysql.connect(
#     host='localhost',
#     user='root',
#     password='',
#     db='e_store',
#     charset='utf8mb4',
#     cursorclass=pymysql.cursors.DictCursor
# )

# Load the dataset into a pandas DataFrame
# query = "SELECT `count`, `pname`, `pbrand`, `status` FROM `tbl_checkout`"
train_data = pd.read_csv('demand1.csv')

label_encoder = LabelEncoder()
train_data['pname'] = label_encoder.fit_transform(train_data['pname'])
train_data['pbrand'] = label_encoder.fit_transform(train_data['pbrand'])

X_train, X_test, y_train, y_test = train_test_split(train_data[['pname', 'pbrand', 'count']], train_data['status'],
                                                    test_size=0.2, random_state=42)
import matplotlib.pyplot as plt

# Plot X_train
plt.scatter(X_train['pname'], X_train['count'], color='blue', label='Train')

# Plot X_test
plt.scatter(X_test['pname'], X_test['count'], color='red', label='Test')

# Add legend and labels
plt.legend()
plt.xlabel('Product Name')
plt.ylabel('Count')

# Show the plot
plt.show()


# Plot
# import seaborn as sns
#
# sns.pairplot(train_data, x_vars=['pname', 'pbrand', 'count'], y_vars='status')

# Create and fit a logistic regression model
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
        update_query = 'UPDATE tbl_checkout SET status = 0'
        cursor.execute(update_query)

        update_query1 = 'UPDATE tbl_checkout SET status = 1 WHERE pname IN ({})'.format(
            ', '.join(['%s'] * len(top_products)))
        cursor.execute(update_query1, top_products)
        test_data['status']=1
        # Prepare the test data and predict the labels
        X_test = test_data[['pname', 'pbrand', 'count']]
        y_test = test_data['status']
        model.fit(X_test, y_test)
        y_pred = model.predict(X_test)

        # Calculate the accuracy of the model on the test data
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)
        print('Predictions:', y_pred)

        # Print the top 10 products with the highest demand
        print("Top 10 products with the highest demand:")
        for product in top_products:
            print(product)

        break

cnx.commit()
# # Close the cursor and connection

cursor.close()
cnx.close()
