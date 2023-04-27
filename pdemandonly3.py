import pandas as pd
import mysql.connector
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset into a pandas DataFrame
train_data = pd.read_csv('demand1.csv')

label_encoder = LabelEncoder()
train_data['pname'] = label_encoder.fit_transform(train_data['pname'])
train_data['pbrand'] = label_encoder.fit_transform(train_data['pbrand'])

X_train, X_test, y_train, y_test = train_test_split(train_data[['pname', 'pbrand', 'count']], train_data['status'],
                                                    test_size=0.2, random_state=42)

# Create and fit a decision tree regressor model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Connect to the database
cnx = mysql.connector.connect(user='root', password='', host='localhost', database='e_store')

# Create a cursor
cursor = cnx.cursor()

# Execute a SELECT statement to retrieve data for products with demand
query = 'SELECT pname, SUM(count) AS total_count FROM tbl_checkout WHERE status=0 GROUP BY pname ORDER BY total_count DESC '
cursor.execute(query)
rows = cursor.fetchall()

# Iterate through the rows and print the product names and total count
for row in rows:
    print(row[0], row[1])

# Update the status of the products with the highest demand
update_query = 'UPDATE tbl_checkout SET status=1 WHERE pname IN (' + ','.join(['%s']*len(rows)) + ')'
cursor.execute(update_query, [row[0] for row in rows])

# Set the status of other products to 0
update_query = 'UPDATE tbl_checkout SET status=0 WHERE pname NOT IN (' + ','.join(['%s']*len(rows)) + ')'
cursor.execute(update_query, [row[0] for row in rows])

cnx.commit()

# Close the cursor and connection
cursor.close()
cnx.close()
