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
    # rest of the code here
    # Iterate through the rows and print the data
    for row in rows:
        # print(row)

        test_data = pd.DataFrame(rows, columns=['pname', 'pbrand', 'count'])
        ###################################
        test_data['pname'] = label_encoder.fit_transform(test_data['pname'])
        test_data['pbrand'] = label_encoder.fit_transform(test_data['pbrand'])
        # Find the product with the highest demand by iterating through the demand_sums dictionary and keeping track of the highest demand and corresponding product ID
        demand_sums = {}

        query1 = 'SELECT pname,count FROM tbl_checkout'
        cursor.execute(query1)
        rows1 = cursor.fetchall()

        for row1 in rows1:
            if row1[0] in demand_sums:
                demand_sums[row1[0]] += row1[1]
            else:
                demand_sums[row1[0]] = row1[1]

        # Find the product with the highest demand by iterating through the demand_sums dictionary and keeping track of the highest demand and corresponding product ID
        max_demand = 0
        max_product_id = None
        for pname, count in demand_sums.items():
            if count > max_demand:
                max_demand = count
                max_product_id = pname
        print(max_product_id)

        # Check if max_demand is greater than 0 and assign the corresponding status to the test_data['status'] column

            # test_data['status'] = 'Highest demand is for product ID ' + str(max_product_id)
        test_data['status'] = 1


        X_test = test_data[['pname', 'pbrand', 'count']]
        y_test = test_data['status']
        # # Predict the accident label on the test data
        model.fit(X_test, y_test)

        y_pred = model.predict(X_test)

        # # Calculate the accuracy of the model on the test data
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)

        print('Predictions:', y_pred)

        block_executed = False

        for prediction in y_pred:
            if prediction:
                if not block_executed:
                    print("Product With Highest Demand")

                    update_query = 'UPDATE tbl_checkout SET status = 1 WHERE pname = %s'
                    cursor.execute(update_query, (max_product_id,))
                    update_query1 = 'UPDATE tbl_checkout SET status =0 WHERE pname!=%s'

                    cursor.execute(update_query1, (max_product_id,))
                    block_executed = True
            else:
                if not block_executed:
                    print("Normal Sales")
                    update_query = 'UPDATE tbl_checkout SET status =0'
                    cursor.execute(update_query)
                    block_executed = True



cnx.commit()
# # Close the cursor and connection

cursor.close()
cnx.close()
