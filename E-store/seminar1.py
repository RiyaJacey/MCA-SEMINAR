import pandas as pd
import pymysql.cursors
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
query = "SELECT `lname`, `lprice`, `loffer`, `ltprice` FROM `tbl_laptop`"
df = pd.read_sql_query(query, connection)

# Encode the `pname` column using label encoding
label_encoder = LabelEncoder()
df['lname'] = label_encoder.fit_transform(df['lname'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['lprice', 'loffer', 'ltprice']], df['lname'], test_size=0.2, random_state=42)

# Create an instance of the DecisionTreeRegressor class
model = DecisionTreeRegressor()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)
y_pred = y_pred.astype(int)

# Decode the predicted values back to their original labels
predicted_pnames = label_encoder.inverse_transform(y_pred)

# Add the predicted sales to the test data
X_test['Predicted Sales'] = predicted_pnames

# Sort the test data by predicted sales in descending order
sorted_data = X_test.sort_values(by=['Predicted Sales'], ascending=False)

# Get the Product Name with the highest predicted demand
lname = sorted_data.iloc[0]['Predicted Sales']

print("Product Name with the highest predicted demand:", lname)

# Close the database connection
connection.close()
