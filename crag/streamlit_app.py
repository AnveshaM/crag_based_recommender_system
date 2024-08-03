import os
import re
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient, models
from tavily import TavilyClient

# Set API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"
groq_client = Groq()
tavily_api_key = os.environ["TAVILY_API_KEY"]

# Initialize Qdrant client
client = QdrantClient(host='localhost', port=6333)
collection_name = "customer_recommendations"


# Function to fetch external data using Tavily API
def fetch_external_data(query):
    tavily_client = TavilyClient(api_key=tavily_api_key)
    external_context = tavily_client.get_search_context(query=query)
    return external_context


# Function to parse the query and extract customer ID
def extract_customer_id(query):
    pattern = r"(?:customer\s*id|customer\s*number|account\s*id)\s*[:#-]?\s*(\d+)"
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(1)
    fallback_match = re.search(r"\b\d{5,}\b", query)
    if fallback_match:
        return fallback_match.group(0)
    return None


# Function to generate the final response using Groq's API for customer-based data
def generate_final_response(query, external_context, target_customer_transactions, similar_customers):
    customer_id = extract_customer_id(query)
    if similar_customers:
        target_articles = {trans['StockCode'] for trans in target_customer_transactions}
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
            for article in new_articles[:10]
        )
    else:
        existing_articles = ""
        similar_user_articles = external_context

    if external_context is None:
        prompt_template = """
            User {customer_id} has interacted with the following articles: {existing_articles}.
            Based on similar users, suggest 3 new articles that User has not interacted with yet.
            Also, provide a reason for each recommendation.

            Here are some articles that similar users have interacted with:
            {similar_user_articles}
            Please provide your recommendations and reasons.
            """
        prompt = prompt_template.format(
            customer_id=customer_id,
            existing_articles=existing_articles,
            similar_user_articles=similar_user_articles
        )
    else:
        prompt_template = """
        User {customer_id} has interacted with the following articles: {existing_articles}.
        Based on similar users and external data, suggest 3 new articles that User has not interacted with yet.
        Also, provide a reason for each recommendation.

        Here are some articles that similar users have interacted with and external search results:
        {similar_user_articles}
        {external_context}
        Please provide your recommendations and reasons.
        """
        prompt = prompt_template.format(
            customer_id=customer_id,
            existing_articles=existing_articles,
            similar_user_articles=similar_user_articles,
            external_context=external_context
        )

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error getting response from Groq: {e}")


# Function to generate the final response using Groq's API for external data
def generate_final_response_from_external(query, external_context):
    prompt_template = """
    Based on external data sources, suggest 2 top recommendations for the following query: "{query}".
    Also, provide a reason for each recommendation.

    External data sources provide the following context:
    {external_context}
    Please provide your recommendations and reasons.
    """
    prompt = prompt_template.format(
        query=query,
        external_context=external_context
    )

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error getting response from Groq: {e}")


# Function to generate recommendations based on the query
def get_customer_recommendations(query):
    customer_id = extract_customer_id(query)
    if not customer_id:
        st.write("✨Fetching recommendations✨")
        external_context = fetch_external_data(query)
        generate_final_response_from_external(query, external_context)
        return
    # st.write(f"Extracted Customer ID: {customer_id}")

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
        st.error(f"Error retrieving customer data: {e}")
        return

    if result[0]:
        st.write(f"✨Generating recommendations for CustomerID {customer_id} ✨")
    else:
        st.write(f"CustomerID {customer_id} not in the database. Using external web search for recommendations.")
        external_context = fetch_external_data(query)
        generate_final_response_from_external(query, external_context)
        return

    target_customer_transactions = result[0][0].payload.get('Transactions', [])

    try:
        similar_customers = client.search(
            collection_name=collection_name,
            query_vector=result[0][0].vector,
            limit=5
        )
        st.write(f"Found {len(similar_customers)} similar customers.")
    except Exception as e:
        st.error(f"Error searching for similar customers: {e}")
        return

    low_similarity = all(res.score < 0.6 for res in similar_customers[1:])
    if low_similarity:
        st.write("Using web search to augment retrieval 🔍")
        external_context = fetch_external_data(query)
        generate_final_response(query, external_context, target_customer_transactions, similar_customers)
    else:
        generate_final_response(query, None, target_customer_transactions, similar_customers)


# Streamlit UI
st.title("Customer Recommendations")

if 'query' not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Enter your query:", st.session_state.query)

if st.button("Get Recommendations"):
    st.session_state.query = query
    get_customer_recommendations(query)
