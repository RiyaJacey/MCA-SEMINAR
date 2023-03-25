import pandas as pd
import pymysql.cursors
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
query = "SELECT `pid`, `pname`, `price`, `offer`, `tprice` FROM `tbl_productdetail`"
df = pd.read_sql_query(query, connection)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['pid','price', 'offer', 'tprice']], df['pid'], test_size=0.2, random_state=42)

# Create an instance of the DecisionTreeRegressor class
model = DecisionTreeRegressor()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Add the predicted sales to the test data
X_test['Predicted Sales'] = y_pred

# Sort the test data by predicted sales in descending order
sorted_data = X_test.sort_values(by=['Predicted Sales'], ascending=False)

# Get the Product ID with the highest predicted demand
pname = sorted_data.iloc[0]['pid']

print("Product ID with the highest predicted demand:", pname)

# Close the database connection
connection.close()
