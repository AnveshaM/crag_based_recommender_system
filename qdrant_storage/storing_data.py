import json
from qdrant_client import QdrantClient
from qdrant_client import models
import pandas as pd
import ast

# Load the JSON data
with open('/Users/anveshamishra/Documents/GitHub/crag_based_recommender_system/data_preprocessing/data_grouped_by_customer_reduced.json', 'r') as f:
    data_by_customer = json.load(f)

user_interaction = pd.read_csv('/Users/anveshamishra/Documents/GitHub/crag_based_recommender_system/user_interaction_data.csv')
index = user_interaction['CustomerID'].tolist()
# Convert the StockCode column from string to list of integers
def convert_to_list_of_ints(stockcode_str):
    try:
        # Convert string representation of list to an actual list of integers
        return [int(i) for i in ast.literal_eval(stockcode_str)]
    except (ValueError, SyntaxError):
        return []

user_interaction['StockCode'] = user_interaction['StockCode'].apply(convert_to_list_of_ints)

# print(user_interaction['StockCode'][0])

# Initialize the Qdrant client (adjust host and port as needed)
client = QdrantClient(host='localhost', port=6333)

# Define the collection name
collection_name = "customer_recommendations"

# Check if the collection exists, delete if it does, and create a new one
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)


# Create a collection

first_collection = client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=250, distance=models.Distance.COSINE)
)


# Index the data into Qdrant
vector_insert = client.upsert(
    collection_name=collection_name,
    points=models.Batch(
        ids=index,
        vectors=user_interaction['StockCode'].tolist()
    )
)
print(vector_insert)

# Initialize an empty list to store aggregated payloads
payloads = []

# Populate the payload list with aggregated data for each customer
for customer_id, transactions in data_by_customer.items():
    aggregated_payload = {
        "CustomerID": customer_id,
        "Transactions": transactions
    }
    payloads.append(aggregated_payload)

# Ensure ids and payloads match in number
assert len(index) == len(payloads), "Number of IDs and payloads must match."

# Add payloads to the indexed data
payloads_insert = client.upsert(
    collection_name=collection_name,
    points=models.Batch(
        ids=index,
        vectors=user_interaction['StockCode'].tolist(),
        payloads=payloads
    )
)

print("Data successfully indexed with payloads.")
