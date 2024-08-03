from qdrant_client import QdrantClient
from qdrant_client import models
# Initialize the Qdrant client (adjust host and port as needed)
client = QdrantClient(host='localhost', port=6333)
# Define the collection name
collection_name = "customer_recommendations"
target_customer_id = "12357"
result = client.scroll(
    collection_name=collection_name,
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(key="CustomerID", match=models.MatchValue(value=target_customer_id)),
        ]
    ),
    limit=1,
    with_payload=True,
    with_vectors=True,
)



# Extract the vector from the result
if result and result[0][0].vector is not None:
    target_vector = result[0][0].vector
    #print(f"Vector for CustomerID {target_customer_id}: {target_vector}")
else:
    #print(f"No vector found for CustomerID {target_customer_id}")
    exit()

# Now you can use this vector to search for similar vectors
similar_customers = client.search(
    collection_name=collection_name,
    query_vector=target_vector,
    limit=2  # You can set the limit to any number you prefer
)

# Print the results
for res in similar_customers:
    print(f"CustomerID: {res.id}, Score: {res.score}")

# Print the results and their payloads
for res in similar_customers:
    customer_id = res.id
    score = res.score
    payload = res.payload
    print(f"CustomerID: {customer_id}, Score: {score}")
    print("Payload:")
    for transaction in payload.get('Transactions', []):
        print(f"  - InvoiceNo: {transaction['InvoiceNo']}, "
              f"StockCode: {transaction['StockCode']}, "
              f"Description: {transaction['Description']}, "
              f"Quantity: {transaction['Quantity']}, "
              f"InvoiceDate: {transaction['InvoiceDate']}, "
              f"UnitPrice: {transaction['UnitPrice']}, "
              f"Country: {transaction['Country']}")