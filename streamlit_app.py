import google.generativeai as genai
import os
import pandas as pd
import numpy as np
import faiss
import time
from dotenv import load_dotenv
import logging
import requests
from bs4 import BeautifulSoup
import zipfile
import shutil
import streamlit as st # Import Streamlit
import pickle # For saving/loading index if using that strategy

# --- Configuration --- (Keep your existing config sections)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
# Use Streamlit secrets if deploying, otherwise keep getenv for local
if not API_KEY:
    try:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
    except:
         st.error("GOOGLE_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
         st.stop()


genai.configure(api_key=API_KEY)

ONET_DOWNLOAD_PAGE = "https://www.onetcenter.org/database.html"
ONET_DATA_DIR = "./onet_data_extracted/"
ONET_ZIP_FILENAME = "onet_database.zip"

# --- O*NET File Configuration --- (Keep your existing ONET_FILES_CONFIG)
ONET_FILES_CONFIG = {
    # --- Base Table ---
    "Occupation Data.txt": {
        "columns": ['O*NET-SOC Code', 'Title', 'Description'],
        "is_base": True,
        "merge_key": 'O*NET-SOC Code'
    },
     "Abilities.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Abilities', "merge_key": 'O*NET-SOC Code', "agg_separator": ", "
    },
    "Alternate Titles.txt": {
        "columns": ['O*NET-SOC Code', 'Alternate Title', 'Short Title', 'Source(s)'],
        "agg_column": 'Alternate Title', "name_prefix": 'Alternate_Titles', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Education, Training, and Experience.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Category', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Education_Training_Experience_Elements', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Emerging Tasks.txt": {
        "columns": ['O*NET-SOC Code', 'Task', 'Category', 'Original Task ID', 'Date Added', 'Source(s)'],
        "agg_column": 'Task', "name_prefix": 'Emerging_Tasks', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Interests.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Interests', "merge_key": 'O*NET-SOC Code', "agg_separator": ", "
    },
    "Job Zones.txt": {
        "columns": ['O*NET-SOC Code', 'Job Zone', 'Date', 'Domain Source'],
        "merge_key": 'O*NET-SOC Code', "direct_merge_cols": ['Job Zone']
    },
    "Knowledge.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Knowledge', "merge_key": 'O*NET-SOC Code', "agg_separator": ", "
    },
     "Related Occupations.txt": {
         "columns": ['O*NET-SOC Code', 'Related O*NET-SOC Code', 'Related O*NET-SOC Code Title', 'Index'], # Assuming title col name
         "agg_column": 'Related O*NET-SOC Code Title', # Aggregate titles of related jobs
         "name_prefix": 'Related_Occupations',
         "merge_key": 'O*NET-SOC Code',
         "agg_separator": " | "
         # Note: This can create very long lists.
    },
     "Sample of Reported Titles.txt": {
        "columns": ['O*NET-SOC Code', 'Reported Job Title', 'Shown Where'],
        "agg_column": 'Reported Job Title', "name_prefix": 'Sample_Reported_Titles', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Skills.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Skills', "merge_key": 'O*NET-SOC Code', "agg_separator": ", "
    },
    "Task Statements.txt": {
        "columns": ['O*NET-SOC Code', 'Task ID', 'Task', 'Task Type', 'Incumbents Responding'],
        "agg_column": 'Task', "name_prefix": 'Tasks', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Technology Skills.txt": {
        "columns": ['O*NET-SOC Code', 'Scale ID', 'Category', 'Example', 'Commodity Code'],
        "agg_column": 'Example', "name_prefix": 'Technology_Examples', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Tools Used.txt": { # Included for completeness, may overlap with T&T
        "columns": ['O*NET-SOC Code', 'Tool or Technology'],
        "agg_column": 'Tool or Technology', "name_prefix": 'Tools_Used_Legacy', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Tools & Technology.txt": {
        "columns": ['O*NET-SOC Code', 'T2 Type', 'T2 Example', 'Commodity Code', 'Hot Technology'],
        "agg_column": 'T2 Example', "name_prefix": 'Tools_Technology', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Work Activities.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Work_Activities', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Work Context.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Category', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Work_Context', "merge_key": 'O*NET-SOC Code', "agg_separator": " | "
    },
    "Work Styles.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Work_Styles', "merge_key": 'O*NET-SOC Code', "agg_separator": ", "
    },
    "Work Values.txt": {
        "columns": ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value'],
        "agg_column": 'Element Name', "name_prefix": 'Work_Values', "merge_key": 'O*NET-SOC Code', "agg_separator": ", "
    },
     # --- Link Tables (Connecting different element IDs) ---
    # These link IDs (e.g., Ability ID to Work Activity ID). Merging them directly based
    # on O*NET-SOC Code is generally not meaningful for the occupation description document.
    # The script will likely skip merging these unless O*NET-SOC Code is present AND
    # a specific (potentially less useful) agg_column is defined.
    "Abilities to Work Activities.txt": {
        "columns": ['O*NET-SOC Code', 'Ability Element ID', 'Work Activity Element ID', 'Scale ID', 'Data Value'], # Verify column names
        "merge_key": 'O*NET-SOC Code',
        # No agg_column defined - merging raw IDs is not useful for text context.
        # Could potentially aggregate 'Work Activity Element ID' but would need mapping to names.
    },
    "Abilities to Work Context.txt": {
        "columns": ['O*NET-SOC Code', 'Ability Element ID', 'Work Context Element ID', 'Scale ID', 'Data Value'], # Verify column names
        "merge_key": 'O*NET-SOC Code',
        # No agg_column defined.
    },
    "Skills to Work Activities.txt": {
        "columns": ['O*NET-SOC Code', 'Skill Element ID', 'Work Activity Element ID', 'Scale ID', 'Data Value'], # Verify column names
        "merge_key": 'O*NET-SOC Code',
        # No agg_column defined.
    },
    "Skills to Work Context.txt": {
         "columns": ['O*NET-SOC Code', 'Skill Element ID', 'Work Context Element ID', 'Scale ID', 'Data Value'], # Verify column names
         "merge_key": 'O*NET-SOC Code',
         # No agg_column defined.
    },
    "Tasks to DWAs.txt": {
        "columns": ['O*NET-SOC Code', 'Task ID', 'DWA ID', 'Date Accepted', 'Source of DWA'], # Verify column names
        "merge_key": 'O*NET-SOC Code',
        "agg_column": 'DWA ID', # Aggregate DWA IDs? Less useful without names.
        "name_prefix": 'Task_Linked_DWA_IDs',
        "agg_separator": ", "
        # Better approach needs joining with DWA Reference first.
    },

    # --- Reference Tables (Defining codes, scales, elements) ---
    # These tables define identifiers used elsewhere. They generally DO NOT contain
    # O*NET-SOC Code and are not suitable for merging onto the occupation document.
    # The script will skip merging these as they lack the merge key or sensible config.
    "Basic Interests to RIASEC.txt": { # Example: Maps Interest IDs to R/I/A/S/E/C codes
         "columns": ['Element ID', 'RIASEC'], # Verify names
         # No O*NET-SOC Code, cannot merge directly.
    },
    "Content Model Reference.txt": { # Defines elements like Knowledge, Skills, Abilities etc.
         "columns": ['Element ID', 'Element Name', 'Description'], # Verify names
         # No O*NET-SOC Code.
    },
    "DWA Reference.txt": { # Defines Detailed Work Activities
         "columns": ['DWA ID', 'DWA Title', 'IWA ID'], # Verify names
         # No O*NET-SOC Code.
    },
    "Education, Training, and Experience Categories.txt": { # Defines Category IDs used in ETE file
         "columns": ['Category', 'Category Description'], # Verify names
         # No O*NET-SOC Code.
    },
    "Interests Illustrative Activities.txt": { # Example activities for Interest areas
        "columns": ['Element ID', 'Illustrative Activity'], # Verify names
        # No O*NET-SOC Code. Could potentially link via Element ID if needed elsewhere.
    },
    "Interests Illustrative Occupations.txt": { # Example occupations for Interest areas
        "columns": ['Element ID', 'O*NET-SOC Code', 'O*NET-SOC Code Title'], # Verify names
        # Contains O*NET-SOC Code, but links Interest -> Occ, not Occ -> Interest info.
    },
    "IWA Reference.txt": { # Defines Intermediate Work Activities
        "columns": ['IWA ID', 'IWA Title', 'Work Activity ID'], # Verify names
        # No O*NET-SOC Code.
    },
    "Job Zone Reference.txt": { # Defines Job Zones
        "columns": ['Job Zone', 'Name', 'Experience', 'Education', 'Job Training', 'Examples', 'SVP Range'], # Verify names
        # No O*NET-SOC Code. (Job Zones file links Occ Code -> Job Zone #)
    },
    "Level Scale Anchors.txt": { # Defines scale anchors (e.g., "Level 1 = Basic")
        "columns": ['Scale ID', 'Anchor Value', 'Anchor Description'], # Verify names
        # No O*NET-SOC Code.
    },
    "Occupation Level Metadata.txt": { # Metadata about survey responses per item per occ
        "columns": ['O*NET-SOC Code', 'Item', 'Response', 'N', 'Standard Error', 'Lower CI Bound', 'Upper CI Bound', 'Date', 'Domain Source'],
        "merge_key": 'O*NET-SOC Code',
        # Aggregating 'Item' or 'Response' isn't very useful text.
        # Merging 'N' (sample size) might be possible but complex due to multiple items. Skipping merge for simplicity.
    },
    "RIASEC Keywords.txt": { # Keywords for RIASEC types
        "columns": ['RIASEC', 'Keyword'], # Verify names
        # No O*NET-SOC Code.
    },
    "Scales Reference.txt": { # Defines all measurement scales
        "columns": ['Scale ID', 'Scale Name', 'Minimum', 'Maximum'], # Verify names
        # No O*NET-SOC Code.
    },
     "Survey Booklet Locations.txt": { # Survey administration metadata
        "columns": ['Scale ID', 'Survey Item Stem', 'Survey Key', 'Booklet Location', 'Item Type'], # Verify names
        # No O*NET-SOC Code.
    },
     "Task Categories.txt": { # Defines task categories (e.g., Core, Supplemental)
        "columns": ['Scale ID', 'Category', 'Category Description', 'Uncommon Task'], # Verify names
        # No O*NET-SOC Code. Task Statements file uses the Category ID/code.
    },
    "Task Ratings.txt": { # Provides ratings for tasks linked to occupations
        "columns": ['O*NET-SOC Code', 'Task ID', 'Scale ID', 'Category', 'Data Value', 'N', 'Standard Error', 'Lower CI Bound', 'Upper CI Bound', 'Date', 'Domain Source'],
        "merge_key": 'O*NET-SOC Code',
        "agg_column": 'Task ID', # Aggregate Task IDs? Less useful. Needs join w/ Task Statements.
        "name_prefix": 'Rated_Task_IDs',
        "agg_separator": ", "
        # Merging actual ratings (Data Value) based on Scale ID needs more complex logic.
    },
    "UNSPSC Reference.txt": { # Defines commodity codes used in T&T
        "columns": ['Commodity Code', 'Commodity Title', 'Class Code', 'Class Title', 'Family Code', 'Family Title', 'Segment Code', 'Segment Title'], # Verify names
        # No O*NET-SOC Code. T&T file uses the Commodity Code.
    },
    "Work Context Categories.txt": { # Defines categories used in Work Context
        "columns": ['Element ID', 'Category', 'Category Description'], # Verify names
        # No O*NET-SOC Code. Work Context file uses the Category ID/code.
    }
}


# Model names
EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = "models/gemini-1.5-flash"

# --- Helper Functions (Download, Load, Merge, Embed, Index, Generate) ---
# Keep ALL your existing helper functions here:
# - download_and_extract_onet(...)
# - load_and_merge_onet_data(...)
# - create_documents(...)
# - embed_texts(...)
# - build_faiss_index(...)
# - find_similar_documents(...)
# - generate_answer(...)
# - Optional: save_index_and_metadata / load_index_and_metadata if using pre-built index

# --- [PREVIOUS SCRIPT's HELPER FUNCTIONS GO HERE] ---
# ... (copy all your existing function definitions) ...
# Example placeholder:
def download_and_extract_onet(page_url, download_dir, zip_filename):
     logging.info("Pretending to download/extract...") # Replace with your actual function
     # Ensure the directory exists for the loading step
     os.makedirs(download_dir, exist_ok=True)
     # In a real scenario, check if files exist before returning True
     # This placeholder assumes data is already there if dir exists.
     return os.path.isdir(download_dir) and len(os.listdir(download_dir)) > 0

def load_and_merge_onet_data(data_dir, files_config):
     logging.info("Pretending to load/merge data...") # Replace with your actual function
     # This needs to return your actual merged_df
     # Placeholder return:
     # return pd.DataFrame({'O*NET-SOC Code': ['1-1'], 'Title': ['Test Occ'], 'Description': ['Desc']})
     # --- PASTE YOUR REAL FUNCTION HERE ---
     pass # Remove this line when pasting

def create_documents(df):
    logging.info("Pretending to create documents...") # Replace with your actual function
    # This needs to return your actual documents, doc_metadata
    # Placeholder return:
    # return ["Doc 1 text"], [{'title': 'Test Occ', 'code': '1-1', 'original_text': 'Doc 1 text'}]
    # --- PASTE YOUR REAL FUNCTION HERE ---
    pass # Remove this line when pasting

def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=100):
    logging.info("Pretending to embed texts...") # Replace with your actual function
    # This needs to return your actual embeddings, valid_original_indices
    # Placeholder return (correct dimension is important - 768 for embedding-001):
    # dummy_embeddings = [list(np.random.rand(768))] * len(texts)
    # return dummy_embeddings, list(range(len(texts)))
    # --- PASTE YOUR REAL FUNCTION HERE ---
    pass # Remove this line when pasting


def build_faiss_index(embeddings):
    logging.info("Pretending to build FAISS index...") # Replace with your actual function
    # This needs to return your actual faiss_index
    # Placeholder return:
    # if not embeddings: return None
    # dim = len(embeddings[0])
    # index = faiss.IndexFlatL2(dim)
    # index.add(np.array(embeddings).astype('float32'))
    # return index
    # --- PASTE YOUR REAL FUNCTION HERE ---
    pass # Remove this line when pasting


def find_similar_documents(query, index, query_embedding_model, doc_metadata_filtered, top_k=4):
    logging.info(f"Pretending to find similar docs for: {query}") # Replace with your actual function
    # This needs to return retrieved_docs (list of strings)
    # Placeholder return:
    # if not doc_metadata_filtered: return []
    # return [doc_metadata_filtered[0]['original_text']] # Just return first doc context
    # --- PASTE YOUR REAL FUNCTION HERE ---
    pass # Remove this line when pasting


def generate_answer(query, context_docs, generation_model_name):
    logging.info(f"Pretending to generate answer for: {query}") # Replace with your actual function
    # This needs to return the LLM's response string
    # Placeholder return:
    # return f"Based on limited context, the answer to '{query}' might involve: {context_docs[0][:100]}..."
    # --- PASTE YOUR REAL FUNCTION HERE ---
    pass # Remove this line when pasting

# --- Streamlit Caching for Resource Loading ---
@st.cache_resource(show_spinner="Loading O*NET data and building index...")
def load_onet_resources():
    """Downloads data, loads, merges, embeds, and builds the index."""
    logging.info("--- Attempting to load O*NET resources ---")

    # --- Step 0: Download and Extract Data (if needed) ---
    if not download_and_extract_onet(ONET_DOWNLOAD_PAGE, ONET_DATA_DIR, ONET_ZIP_FILENAME):
        logging.critical("Failed to download or extract O*NET data. Cannot proceed.")
        st.error("Fatal Error: Could not download or find O*NET data files.")
        return None, None # Return None to indicate failure

    # --- Step 1: Load and Merge Data ---
    logging.info(f"Loading and merging O*NET data from: {ONET_DATA_DIR} based on configuration.")
    merged_onet_data = load_and_merge_onet_data(ONET_DATA_DIR, ONET_FILES_CONFIG)
    if merged_onet_data is None or merged_onet_data.empty:
        logging.error("Failed to load or merge O*NET data.")
        st.error("Error: Failed to load or merge O*NET data files. Check logs and configuration.")
        return None, None

    # --- Step 2: Create Document Chunks ---
    logging.info("Creating document chunks from merged data...")
    all_documents, all_doc_metadata = create_documents(merged_onet_data)
    if not all_documents:
         logging.error("No documents were created.")
         st.error("Error: No documents generated from O*NET data. Check data processing steps.")
         return None, None

    # --- Step 3: Embed Documents ---
    logging.info("Embedding documents...")
    doc_embeddings, valid_original_indices = embed_texts(all_documents)
    if not doc_embeddings:
        logging.error("No documents were successfully embedded.")
        st.error("Error: Failed to embed O*NET documents. Check API key and embedding service.")
        return None, None

    # --- Filter Metadata to match successful embeddings ---
    filtered_doc_metadata = [all_doc_metadata[i] for i in valid_original_indices]
    logging.info(f"Proceeding with {len(filtered_doc_metadata)} successfully embedded documents.")

    # --- Step 4: Build FAISS Index ---
    logging.info("Building FAISS index...")
    faiss_index = build_faiss_index(doc_embeddings)
    if faiss_index is None:
        logging.error("Failed to build the search index.")
        st.error("Error: Failed to build the search index for O*NET data.")
        return None, None

    logging.info("--- O*NET resources loaded successfully ---")
    return faiss_index, filtered_doc_metadata

# --- Streamlit App UI ---

st.title("O*NET Career Chatbot")
st.write("Ask questions about occupations based on the O*NET database.")

# --- Load Resources using Cache ---
# This will only run the function once per session or if the code changes
faiss_index, filtered_doc_metadata = load_onet_resources()

# --- Check if loading failed ---
if faiss_index is None or filtered_doc_metadata is None:
    st.error("Initialization failed. Cannot start chatbot. Please check the logs.")
    st.stop() # Stop execution if resources aren't loaded

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat messages from history on app rerun ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- React to user input ---
if prompt := st.chat_input("Ask about an occupation..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display "thinking" spinner while processing
    with st.spinner("Bot: Thinking..."):
        try:
            # 1. Retrieve relevant documents
            retrieved_context = find_similar_documents(
                prompt,
                faiss_index,
                EMBEDDING_MODEL,
                filtered_doc_metadata,
                top_k=4 # Adjust K as needed
            )

            # 2. Generate answer based on context
            answer = generate_answer(prompt, retrieved_context, GENERATION_MODEL)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            logging.error(error_message)
            st.error(error_message)
            # Optionally add error to chat history
            # st.session_state.messages.append({"role": "assistant", "content": f"Sorry, {error_message}"})
