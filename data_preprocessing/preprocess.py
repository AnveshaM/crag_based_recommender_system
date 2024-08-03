import pandas as pd
import json
import numpy as np


# Load the CSV data into a pandas DataFrame
df = pd.read_csv('/Users/anveshamishra/Downloads/data.csv', encoding='ISO-8859-1')

# Convert CustomerID to integer, handling missing values if necessary
df = df.dropna()
df['CustomerID'] = df['CustomerID'].astype(int)

print("Original dataframe:")
print(df.info())

# Step 1: Remove all the StockCodes which only contain alphabets in their values.
df = df[df['StockCode'].str.isalnum() & df['StockCode'].str.isdigit()]
# Step 2: Create a unique list of StockCodes after filtering
unique_stock_codes = df['StockCode'].unique()
print(f"Unique Stock Codes: {unique_stock_codes}")

# Step 3 : Create a positional mapping for each StockCode
stock_code_mapping = {code: idx+1 for idx, code in enumerate(unique_stock_codes)}
# Save the mapping list to a CSV file
mapping_df = pd.DataFrame(list(stock_code_mapping.items()), columns=['StockCode', 'MappedValue'])
print(mapping_df.head())
mapping_df.to_csv('stock_code_mapping.csv', index=False)

# Step 4: Replace StockCodes with positional mapping
df['StockCode'] = df['StockCode'].map(stock_code_mapping)

#Step 5 : Drop rows where StockCode values are greater than 250
df = df[df['StockCode'] <= 250]
# Define number of users and items
num_customers = len(df['CustomerID'].unique())
print('Total Users:', num_customers)

num_items = len(df['StockCode'].unique())
print('Total Items:', num_items)

# Group the data by CustomerID
grouped = df.groupby('CustomerID')

# Initialize a dictionary to hold the grouped data with sampled readings
data_by_customer = {}

# Set the random seed for reproducibility
np.random.seed(42)

# Iterate over each group
for customer_id, group in grouped:
    # Drop the 'CustomerID' column and convert the rest of the group to a dictionary
    data_by_customer[int(customer_id)] = group.drop(columns=['CustomerID']).to_dict(orient='records')

# Convert the dictionary to a JSON string
json_data = json.dumps(data_by_customer, indent=4)

# Write the JSON data to a file
output_path = '/Users/anveshamishra/Documents/GitHub/crag_based_recommender_system/data_preprocessing/data_grouped_by_customer_reduced.json'
with open(output_path, 'w') as f:
    f.write(json_data)

print(f"Data successfully grouped, sampled, reduced, and saved as JSON at {output_path}.")


df2 = df[['CustomerID','StockCode']]
df2 = df2.groupby(['CustomerID'])['StockCode'].agg(list).reset_index()
df2['StockCode']= df2['StockCode'].transform(lambda x : [0 if y+1 not in x else y+1 for y in range(num_items)])
print(df2.head())
print(df2.info())
filename = 'user_interaction_data.csv'
df2.to_csv(filename,index = False)
print(f"User Interactive CSV file saved successfully as {filename}.")