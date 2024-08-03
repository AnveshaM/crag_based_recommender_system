import os
from groq import Groq
from qdrant_client import QdrantClient, models
import re
from ragas.run_config import RunConfig
from tavily import TavilyClient
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import json
from datasets import Dataset
# from langchain_together import Together
# from langchain_together.embeddings import TogetherEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


# Set API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = "sk-proj-DQ1R3C8pGpIlJiWePLF9T3BlbkFJ3b3icAjYGI7wkPAVGqEq"
groq_client = Groq()
tavily_api_key = os.environ["TAVILY_API_KEY"]
# os.environ["TOGETHER_API_KEY"] = "61eb011da7353b572f7e3d776480d2c285d40050e3c2db034350839d5d44f217"
# together_key = os.environ["TOGETHER_API_KEY"]
# embeddings = TogetherEmbeddings(model="togethercomputer/mtogethercomputer/m2-bert-80M-8k-retrieval")

# together_completion = Together(
#     model="meta-llama/Meta-Llama-3-70B",
#     temperature=0.7,
#     max_tokens=8100,
#     top_k=1,
#     together_api_key=together_key
# )

# define llm and embeddings
langchain_llm = Ollama(model="llama3") # any langchain LLM instance
fast_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en")
# model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
# gpt4all_kwargs = {'allow_download': 'True'}
# embeddings = GPT4AllEmbeddings(
#     model_name=model_name,
#     gpt4all_kwargs=gpt4all_kwargs
# )

# make sure to wrap them with wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# langchain_llm = LangchainLLMWrapper(langchain_llm)




# Initialize Qdrant client
client = QdrantClient(host='localhost', port=6333)
collection_name = "customer_recommendations"

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

def build_context(customer_id=None, existing_articles=None, similar_user_articles=None):
    """Build context string from available data."""
    context_parts = []

    if customer_id:
        context_parts.append(f"Customer ID is {customer_id}")

    if existing_articles:
        context_parts.append(f"Existing Articles are {existing_articles}")

    if similar_user_articles:
        context_parts.append(f"Similar User Articles are {similar_user_articles}")

    return "\n".join(context_parts)


def get_customer_recommendations(query):
    customer_id = extract_customer_id(query)

    if not customer_id:
        print("No customer ID found in the query. Fetching general recommendations.")

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
        return None, None

    # Check if any results were found
    if result[0]:
        print(f"Found vector and data for CustomerID {customer_id}")
    else:
        print(f"No vector found for CustomerID {customer_id}. Using external data for recommendations.")

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

    # Define a prompt template
    prompt_template = """
            User {customer_id} has interacted with the following articles: {existing_articles}.
            Based on similar users and external data, suggest 1 new article that User has not interacted with yet.
            Also, provide a reason for each recommendation.

            Here are some articles that similar users have interacted with and external search results:
            {similar_user_articles}
            Please provide your recommendations and reasons.
            """
    # Build the context
    context = build_context(
        customer_id=customer_id,
        existing_articles=existing_articles,
        similar_user_articles=similar_user_articles
    )

    print(context)
    # Fill the prompt template with data
    prompt = prompt_template.format(
        customer_id=customer_id,
        existing_articles=existing_articles,
        similar_user_articles=similar_user_articles
    )
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192"
    )
    # Extract and return the response content
    recommendations = response.choices[0].message.content
    return recommendations, context

# Example usage and evaluation
test_queries = [
    "recommend me articles for customer id 12357 that it has not yet bought",
    "recommend me similar articles for customer id 18228"
]

ground_truth = [
    "Based on the user's history and external data, I recommend the PIGGY BANK RETROSPOT (StockCode: 77). This item matches the user's interest in RetroSpot-themed products and fits their preference for decorative home decor. Additionally, similar users have shown interest in this item, making it a great choice for User 12357.",
"Based on the user's history and external data, I recommend the JUMBO SHOPPER VINTAGE RED PAISLEY (StockCode: 70). This item suits the user's preference for home decor and novelty items, as evidenced by their interest in hot water bottles and metal signs. The bag's vintage and decorative style, along with its vibrant and whimsical design, aligns with the user's taste for quirky and humorous items, making it a likely favorite."
]
# "recommend me some items to buy if I have recently purchased a SET OF 3 BUTTERFLY COOKIE CUTTERS",
# "recommend me articles for customer id 12556 if the customer is interested in home decor"
# "any baking related tool",
# "any heart shaped decor or any other quirky home decor items"
answers = []
contexts = []


def clean_text(text):
    # Remove any kind of brackets and the content inside them, and commas
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}|,', '', text)

    # Remove newline characters
    text = text.replace('\n', ' ')

    # Remove any special characters except for spaces and letters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

for query in test_queries:
    response, context = get_customer_recommendations(query)
    if response is not None:
        answers.append(clean_text(response))
        contexts.append([clean_text(context)])

# Prepare data for evaluation
data = {
    "question": test_queries,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truth
}
# Convert the dictionary to a JSON string
json_data = json.dumps(data, indent=4)

# Write the JSON data to a file
output_path = '/Users/anveshamishra/Documents/GitHub/crag_based_recommender_system/data_preprocessing/datasetraga.json'
with open(output_path, 'w') as f:
    f.write(json_data)

# Convert dict to dataset
dataset = Dataset.from_dict(data)

# Evaluate the recommendations
result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=langchain_llm,
    embeddings=fast_embeddings,
    raise_exceptions=False,
    run_config=RunConfig(timeout=120.0)
)

# Convert the result to a pandas DataFrame for easier analysis
df = result.to_pandas()
df.to_csv('results.csv')
print(df)


