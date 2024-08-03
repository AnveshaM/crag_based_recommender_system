import os
from groq import Groq
from qdrant_client import QdrantClient, models
import re
from tavily import TavilyClient


# Set API keys
os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
os.environ["TAVILY_API_KEY"] = "TAVILY_API_KEY"
groq_client = Groq()
tavily_api_key = os.environ["TAVILY_API_KEY"]

# Initialize Qdrant client
client = QdrantClient(host='localhost', port=6333)
collection_name = "customer_recommendations"


# Function to fetch external data using Tavily API
def fetch_external_data(query):
    # Simulated external search result as an example
    # Step 1. Instantiating your TavilyClient
    tavily_client = TavilyClient(api_key=tavily_api_key)
    # Step 2. Executing a context search query
    external_context = tavily_client.get_search_context(query=query)
    return external_context





# Function to parse the query and extract customer ID
def extract_customer_id(query):
    # Define a pattern that looks for keywords associated with customer IDs followed by a number
    pattern = r"(?:customer\s*id|customer\s*number|account\s*id)\s*[:#-]?\s*(\d+)"

    # Search for the pattern in the query
    match = re.search(pattern, query, re.IGNORECASE)

    if match:
        return match.group(1)

    # As a fallback, check for any number without specific keywords
    # Avoid numbers that are unlikely to be customer IDs (e.g., very small numbers)
    fallback_match = re.search(r"\b\d{5,}\b", query)  # Assuming IDs are at least 4 digits long
    if fallback_match:
        return fallback_match.group(0)

    return None




# Function to generate the final response using Groq's API for customer-based data
def generate_final_response(query, external_context, target_customer_transactions, similar_customers):
    customer_id = extract_customer_id(query)
    if similar_customers:
        # Extract articles already bought by the target customer
        target_articles = {trans['StockCode'] for trans in target_customer_transactions}

        # Collect articles from similar customers
        new_articles = []
        for res in similar_customers:
            payload = res.payload
            for transaction in payload.get('Transactions', []):
                stock_code = transaction['StockCode']
                if stock_code not in target_articles:
                    new_articles.append(transaction)

        existing_articles = ", ".join(
            f"{trans['Description']} (StockCode: {trans['StockCode']})" for trans in target_customer_transactions)
        similar_user_articles = "\n".join(
            f"Article: {article['Description']}, StockCode: {article['StockCode']}, Details: {article}"
            for article in new_articles[:10]  # Limit to a subset for the prompt
        )
    else:
        existing_articles = ""
        similar_user_articles = external_context
    if external_context is None :
        # Define a prompt template
        prompt_template = """
            User {customer_id} has interacted with the following articles: {existing_articles}.
            Based on similar users, suggest 3 new articles that User has not interacted with yet.
            Also, provide a reason for each recommendation.

            Here are some articles that similar users have interacted with:
            {similar_user_articles}
            Please provide your recommendations and reasons.
            """
        # Fill the prompt template with data
        prompt = prompt_template.format(
            customer_id=customer_id,
            existing_articles=existing_articles,
            similar_user_articles=similar_user_articles
        )
    else :
        # Define a prompt template
        prompt_template = """
        User {customer_id} has interacted with the following articles: {existing_articles}.
        Based on similar users and external data, suggest 3 new articles that User has not interacted with yet.
        Also, provide a reason for each recommendation.
    
        Here are some articles that similar users have interacted with and external search results:
        {similar_user_articles}
        {external_context}
        Please provide your recommendations and reasons.
        """

        # Fill the prompt template with data
        prompt = prompt_template.format(
            customer_id=customer_id,
            existing_articles=existing_articles,
            similar_user_articles=similar_user_articles,
            external_context = external_context
        )
    #print(prompt)

    # Use Groq API to get chat completion
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        # Print the response
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting response from Groq: {e}")


# Function to generate the final response using Groq's API for external data
def generate_final_response_from_external(query, external_context):
    # Define a prompt template for external data
    prompt_template = """
    Based on external data sources, suggest 2 top recommendations for the following query: "{query}".
    Also, provide a reason for each recommendation.

    External data sources provide the following context:
    {external_context}
    Please provide your recommendations and reasons.
    """
    #query_embedding = query_to_embedding(query)
    # Fill the prompt template with data
    prompt = prompt_template.format(
        query=query,
        external_context=external_context
    )
    #print(prompt)

    # Use Groq API to get chat completion
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        # Print the response
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting response from Groq: {e}")


# Function to generate recommendations based on the query
def get_customer_recommendations(query):
    customer_id = extract_customer_id(query)

    if not customer_id:
        print("No customer ID found in the query. Fetching general recommendations.")
        external_context = fetch_external_data(query)
        generate_final_response_from_external(query, external_context)
        return

    print(f"Extracted Customer ID: {customer_id}")

    # Step 1: Find the vector for the target customer
    try:
        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="CustomerID", match=models.MatchValue(value=customer_id)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
    except Exception as e:
        print(f"Error retrieving customer data: {e}")
        return

    # Check if any results were found
    if result[0]:
        print(f"Found vector and data for CustomerID {customer_id}")
    else:
        print(f"No vector found for CustomerID {customer_id}. Using external data for recommendations.")
        external_context = fetch_external_data(query)
        generate_final_response_from_external(query, external_context)
        return

    # Step 2: Extract target customer transactions
    target_customer_transactions = result[0][0].payload.get('Transactions', [])

    # Step 3: Find similar customers
    try:
        similar_customers = client.search(
            collection_name=collection_name,
            query_vector=result[0][0].vector,
            limit=5
        )
        print(f"Found {len(similar_customers)} similar customers.")
    except Exception as e:
        print(f"Error searching for similar customers: {e}")
        return
    for res in similar_customers:
        print(res.score)
        # Check if similarity scores are below a threshold (e.g., 0.1%)
    low_similarity = all(res.score < 0.6 for res in similar_customers[1:])
    print(low_similarity)

    if low_similarity:
        print("Low similarity scores found. Using external data to augment context.")
        external_context = fetch_external_data(query)
        generate_final_response(query, external_context, target_customer_transactions, similar_customers)
    else:
        generate_final_response(query, None, target_customer_transactions, similar_customers)


# Example usage
# query = "recommend me articles for customer id 12535"
# get_customer_recommendations(query)


# Example usage
# query = "recommend me some items to buy if I have recently purchased a SET OF 3 BUTTERFLY COOKIE CUTTERS"
# get_customer_recommendations(query)

# Example usage
# query = "recommend me articles for customer id 12556 if the customer is interested in home decor"
# get_customer_recommendations(query)

# Example usage
# query = "recommend me articles for customer id 12357"
# get_customer_recommendations(query)

# Example usage
# query = "recommend me articles for customer who has previously bought item id 23"
# get_customer_recommendations(query)

# Example usage
query = "recommend me similar articles for customer id 18228"
get_customer_recommendations(query)


