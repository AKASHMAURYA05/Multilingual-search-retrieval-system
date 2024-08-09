import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator
import time

# Load the product dataset
@st.cache_data
def load_data():
    return pd.read_csv('flipkart_com-ecommerce_sample.csv')

df = load_data()

# Preprocess text by converting to lowercase and removing punctuation
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = text.replace('[^\w\s]', '')  # Remove punctuation
        return text
    return ""

# Combine all relevant columns into a single text field
@st.cache_data
def prepare_data(df):
    df['combined'] = (
        df['product_name'].apply(preprocess) + ' ' +
        df['product_category_tree'].apply(preprocess) + ' ' +
        df['description'].apply(preprocess) + ' ' +
        df['brand'].apply(preprocess) + ' ' +
        df['product_specifications'].apply(preprocess)
    )
    return df

df = prepare_data(df)

# Vectorize the combined text using TF-IDF
@st.cache_resource
def create_vectorizer(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = create_vectorizer(df)

# Store the product IDs and names for retrieval
product_ids = df['uniq_id'].tolist()
product_names = df['product_name'].tolist()

# Function to detect and translate the query if it's not in English
def translate_query(query):
    detected_lang = detect(query)
    if detected_lang != 'en':
        translation = GoogleTranslator(source=detected_lang, target='en').translate(query)
        return translation
    return query

# Function to search for products
def search_products(query, top_n=10):
    query = preprocess(query)
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(-cosine_similarities)[:top_n]
    top_products = [(product_ids[i], product_names[i], cosine_similarities[i]) for i in top_indices]
    return top_products

# Custom CSS to enhance the appearance
st.markdown("""
    <style>
    .main {
     background: rgb(184,144,210);
background: linear-gradient(90deg, rgba(184,144,210,0.24271715522146353) 0%, rgba(55,157,162,1) 100%, rgba(252,176,69,1) 100%); 
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
    
        color: #ece1f0;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 30px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextInput input {
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #dddddd;
    }
    .stSlider > div {
        padding: 0 20px;
    }
    .product-card {
     background: rgb(238,174,202);
background: radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 100%); 
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #3498db;
        transition: transform 0.2s ease-in-out;
    }
    .product-card:hover {
        transform: translateY(-5px);
    }
    .product-card h4 {
       margin: 0 0 10px 0;
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .product-card p {
        margin: 5px 0;
        color: #34495e;
    }
    .product-id {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .product-score {
        font-weight: 1000;
        color: #27ae60;
    }
    .metrics {
        margin-top: 20px;
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metrics h4 {
        margin-bottom: 10px;
        color: #2c3e50;
    }
    .metrics p {
        margin: 5px 0;
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app with a sidebar
st.title('🛒 Product Search App with Multilingual Support')

with st.sidebar:
    st.header("Search Settings")
    top_n = st.slider("Number of results to display:", min_value=1, max_value=20, value=10)

query = st.text_input("Enter a search term:")

if st.button('Search'):
    if query:
        start_time = time.time()
        
        # Translation
        translate_start_time = time.time()
        translated_query = translate_query(query)
        translate_time = time.time() - translate_start_time
        
        # Search
        search_start_time = time.time()
        results = search_products(translated_query, top_n)
        search_time = time.time() - search_start_time
        
        # Total time
        total_time = time.time() - start_time
        
        if results:
            st.write(f"Top {len(results)} results for '{query}':")
            
            # Collect all the data of the search results
            result_data = []
            for i, (prod_id, prod_name, score) in enumerate(results, start=1):
                product_details = df[df['uniq_id'] == prod_id].iloc[0]
                
                # Extract and format the specifications
                specifications = product_details['product_specifications']
                if isinstance(specifications, str):
                    specifications = specifications.replace("{", "").replace("}", "").replace("=>", ":").replace("'", "").replace(",", ", ")
                else:
                    specifications = "N/A"
                
                st.markdown(f"""
                <div class="product-card">
                    <h4>{i}. {prod_name}</h4>
                    <p><strong>ID:</strong> {prod_id}</p>
                    <p><strong>SCORE:</strong> {score:.4f}</p>
                    <p><strong>CATEGORY:</strong> {product_details['product_category_tree']}</p>
                    <p><strong>BRAND:</strong> {product_details['brand']}</p>
                    <p><strong>DESCRIPTION:</strong> {product_details['description']}</p>
                    <p><strong>SPECIFICATIONS:</strong> {specifications}</p>
                </div>
                """, unsafe_allow_html=True)
                
            
        else:
            st.write(f"No results found for '{query}'. Please enter a valid search term.")
        
        # Display performance metrics
        st.markdown(f"""
        <div class="metrics">
            <h4>Performance Metrics</h4>
            <p><strong>Total Time:</strong> {total_time:.4f} seconds</p>
            <p><strong>Translation Time:</strong> {translate_time:.4f} seconds</p>
            <p><strong>Search Time:</strong> {search_time:.4f} seconds</p>
            <p><strong>Total Products:</strong> {len(df)} products</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
                <div class="product-card">
                <p>Please enter a search term first</p>
                </div>
                """, unsafe_allow_html=True)
