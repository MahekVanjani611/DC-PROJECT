import google.generativeai as genai
import os
import pandas as pd
import numpy as np
import faiss
import time
from dotenv import load_dotenv # Optional
import logging # Use logging for better messages

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv() # Optional: Load environment variables from .env file
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)

# --- IMPORTANT: Set correct paths ---
ONET_DATA_DIR = "./onet_data/" # CHANGE THIS to the directory containing your O*NET files

# --- O*NET File Configuration ---
# Define which files to load, relevant columns, and how to aggregate (if needed)
# 'agg_column': The column containing the text to combine for a single occupation.
# 'name_prefix': Used to name the combined column (e.g., "Tasks_Combined").
# 'merge_key': Usually 'O*NET-SOC Code'.
# 'columns': List of columns to initially load from the file.
ONET_FILES_CONFIG = {

    # --- Base Table ---
    "Occupation Data.txt": {
        "columns": ['O*NET-SOC Code', 'Title', 'Description'],
        "is_base": True,
        "merge_key": 'O*NET-SOC Code'
    },

    # --- Core Occupation Descriptors (Aggregated/Merged Where Sensible) ---
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
         "columns": ['O*NET-SOC Code','Element ID', 'RIASEC'], # Verify names
         "merge_key":'O*NET-SOC Code',
         # No O*NET-SOC Code, cannot merge directly.
    },
    "Content Model Reference.txt": { # Defines elements like Knowledge, Skills, Abilities etc.
         "columns": ['O*NET-SOC Code','Element ID', 'Element Name', 'Description'], # Verify names
         # No O*NET-SOC Code.
                                     "merge_key":'O*NET-SOC Code',
                                     
                                  
    },
    "DWA Reference.txt": { # Defines Detailed Work Activities
         "columns": ['O*NET-SOC Code','DWA ID', 'DWA Title', 'IWA ID'], # Verify names
         # No O*NET-SOC Code.
                           "merge_key":'O*NET-SOC Code',
    },
    "Education, Training, and Experience Categories.txt": { # Defines Category IDs used in ETE file
         "columns": ['O*NET-SOC Code','Category', 'Category Description'], # Verify names
         # No O*NET-SOC Code.
                                                            "merge_key":'O*NET-SOC Code',
    },
    "Interests Illustrative Activities.txt": { # Example activities for Interest areas
        "columns": ['O*NET-SOC Code','Element ID', 'Illustrative Activity'], # Verify names
        # No O*NET-SOC Code. Could potentially link via Element ID if needed elsewhere.
                                               "merge_key":'O*NET-SOC Code',
    },
    "Interests Illustrative Occupations.txt": { # Example occupations for Interest areas
        "columns": ['O*NET-SOC Code','Element ID', 'O*NET-SOC Code', 'O*NET-SOC Code Title'], # Verify names
        # Contains O*NET-SOC Code, but links Interest -> Occ, not Occ -> Interest info.
                                                "merge_key":'O*NET-SOC Code',

    },
    "IWA Reference.txt": { # Defines Intermediate Work Activities
        "columns": ['O*NET-SOC Code','IWA ID', 'IWA Title', 'Work Activity ID'], # Verify names
        # No O*NET-SOC Code.
                           "merge_key":'O*NET-SOC Code',
    },
    "Job Zone Reference.txt": { # Defines Job Zones
        "columns": ['O*NET-SOC Code','Job Zone', 'Name', 'Experience', 'Education', 'Job Training', 'Examples', 'SVP Range'], # Verify names
        # No O*NET-SOC Code. (Job Zones file links Occ Code -> Job Zone #)
        "merge_key":'O*NET-SOC Code',
    },
    "Level Scale Anchors.txt": { # Defines scale anchors (e.g., "Level 1 = Basic")
        "columns": ['O*NET-SOC Code','Scale ID', 'Anchor Value', 'Anchor Description'], # Verify names
        # No O*NET-SOC Code.
                                 "merge_key":'O*NET-SOC Code',
    },
    "Occupation Level Metadata.txt": { # Metadata about survey responses per item per occ
        "columns": ['O*NET-SOC Code', 'Item', 'Response', 'N', 'Standard Error', 'Lower CI Bound', 'Upper CI Bound', 'Date', 'Domain Source'],
        "merge_key": 'O*NET-SOC Code',
                                       
        # Aggregating 'Item' or 'Response' isn't very useful text.
        # Merging 'N' (sample size) might be possible but complex due to multiple items. Skipping merge for simplicity.
    },
    "RIASEC Keywords.txt": { # Keywords for RIASEC types
        "columns": ['O*NET-SOC Code','RIASEC', 'Keyword'], # Verify names
                             "merge_key":'O*NET-SOC Code',
        # No O*NET-SOC Code.
    },
    "Scales Reference.txt": { # Defines all measurement scales
        "columns": ['O*NET-SOC Code','Scale ID', 'Scale Name', 'Minimum', 'Maximum'], # Verify names
                              "merge_key":'O*NET-SOC Code',
        # No O*NET-SOC Code.
    },
     "Survey Booklet Locations.txt": { # Survey administration metadata
        "columns": ['O*NET-SOC Code','Scale ID', 'Survey Item Stem', 'Survey Key', 'Booklet Location', 'Item Type'], # Verify names
        # No O*NET-SOC Code.
                                       "merge_key":'O*NET-SOC Code',
    },
     "Task Categories.txt": { # Defines task categories (e.g., Core, Supplemental)
        "columns": ['O*NET-SOC Code','Scale ID', 'Category', 'Category Description', 'Uncommon Task'], # Verify names
        # No O*NET-SOC Code. Task Statements file uses the Category ID/code.
                              "merge_key":'O*NET-SOC Code',
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
        "columns": ['O*NET-SOC Code','Commodity Code', 'Commodity Title', 'Class Code', 'Class Title', 'Family Code', 'Family Title', 'Segment Code', 'Segment Title'], # Verify names
        # No O*NET-SOC Code. T&T file uses the Commodity Code.
                              "merge_key":'O*NET-SOC Code',
    },
    "Work Context Categories.txt": { # Defines categories used in Work Context
        "columns": ['O*NET-SOC Code','Element ID', 'Category', 'Category Description'], # Verify names
        # No O*NET-SOC Code. Work Context file uses the Category ID/code.
                                     "merge_key":'O*NET-SOC Code',
    }
}

# Model names
EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = "models/gemini-1.5-flash"

# --- 1. Generalized O*NET Data Loading and Merging ---

def load_and_merge_onet_data(data_dir, files_config):
    """Loads multiple O*NET files based on config and merges them."""
    all_data = {}
    base_df = None
    base_key = 'O*NET-SOC Code' # Assuming this is the primary key

    # --- Load all configured files ---
    for filename, config in files_config.items():
        filepath = os.path.join(data_dir, filename)
        try:
            # Use error handling for encoding, try utf-8 as fallback
            try:
                df = pd.read_csv(filepath, sep='\t', encoding='latin1', on_bad_lines='warn')
            except UnicodeDecodeError:
                logging.warning(f"Latin-1 failed for {filename}, trying UTF-8.")
                df = pd.read_csv(filepath, sep='\t', encoding='utf-8', on_bad_lines='warn')

            # Select only the necessary columns early to save memory
            if 'columns' in config:
                cols_to_keep = [col for col in config['columns'] if col in df.columns]
                # Ensure the merge key is always kept if it exists
                merge_key = config.get('merge_key', base_key)
                if merge_key in df.columns and merge_key not in cols_to_keep:
                     cols_to_keep.append(merge_key)
                df = df[cols_to_keep]

            all_data[filename] = df
            logging.info(f"Successfully loaded {len(df)} rows from {filename}")

            if config.get("is_base"):
                base_df = df
                logging.info(f"Set '{filename}' as the base DataFrame.")

        except FileNotFoundError:
            logging.warning(f"File not found, skipping: {filepath}")
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")

    if base_df is None:
        logging.error("No base DataFrame identified in config (set 'is_base': True for one file). Cannot proceed.")
        return None

    # --- Merge other dataframes onto the base ---
    for filename, df in all_data.items():
        config = files_config[filename]
        if config.get("is_base"):
            continue # Skip merging base with itself

        merge_key = config.get('merge_key', base_key)
        agg_col = config.get('agg_column')

        if merge_key not in base_df.columns:
             logging.warning(f"Base DataFrame missing merge key '{merge_key}' needed for {filename}. Skipping merge.")
             continue
        if merge_key not in df.columns:
             logging.warning(f"DataFrame from {filename} missing merge key '{merge_key}'. Skipping merge.")
             continue

        processed_df_to_merge = None

        # --- Aggregation for one-to-many relationships ---
        if agg_col:
            if agg_col not in df.columns:
                logging.warning(f"Aggregation column '{agg_col}' not found in {filename}. Skipping aggregation.")
                continue

            name_prefix = config.get('name_prefix', filename.split('.')[0]) # Default prefix
            agg_separator = config.get('agg_separator', ' | ')
            combined_col_name = f"{name_prefix}_Combined"

            logging.info(f"Aggregating '{agg_col}' from {filename} into '{combined_col_name}'...")

            # Ensure agg_col is string, handle potential NaN before grouping
            # Use .astype(str) within lambda to handle mixed types gracefully if needed
            df[agg_col] = df[agg_col].fillna('') # Fill NaNs before converting/joining

            # Group by the merge key and aggregate the specified column
            grouped = df.groupby(merge_key)[agg_col].apply(lambda x: agg_separator.join(x.astype(str))).reset_index()
            grouped.rename(columns={agg_col: combined_col_name}, inplace=True)
            processed_df_to_merge = grouped
            fill_value = f"No {name_prefix.lower().replace('_', ' ')} listed."

        # --- Direct Merge (Optional - if needed for one-to-one data) ---
        else:
            # If no aggregation, maybe select specific columns for direct merge?
            # This example focuses on aggregation, so we'll skip direct merge for now.
            # If you had a file with unique info per O*NET code not needing aggregation,
            # you would prepare `df` (e.g., select columns) and set `processed_df_to_merge = df`
            logging.info(f"No aggregation specified for {filename}. Skipping merge in this example.")
            continue # Skip if no aggregation defined for simplicity

        # --- Perform the merge ---
        if processed_df_to_merge is not None:
            try:
                base_df = pd.merge(base_df, processed_df_to_merge, on=merge_key, how='left')
                # Fill NaNs introduced by the left merge only for the newly added combined column
                if combined_col_name in base_df.columns:
                     base_df[combined_col_name].fillna(fill_value, inplace=True)
                logging.info(f"Successfully merged data from {filename}.")
            except Exception as e:
                logging.error(f"Error merging data from {filename}: {e}")


    logging.info(f"Final merged DataFrame has {len(base_df)} rows and columns: {list(base_df.columns)}")
    return base_df


# --- 2. Prepare Documents for RAG (Dynamically uses merged columns) ---

def create_documents(df):
    """Creates text documents from the merged DataFrame for embedding."""
    documents = []
    doc_metadata = [] # To store original info

    # Dynamically find combined columns to include
    combined_cols = [col for col in df.columns if col.endswith('_Combined')]
    logging.info(f"Creating documents using combined columns: {combined_cols}")

    for index, row in df.iterrows():
        # Start with base info
        doc_text_parts = [
            f"Occupation Title: {row.get('Title', 'N/A')}",
            f"O*NET-SOC Code: {row.get('O*NET-SOC Code', 'N/A')}",
            f"Description: {row.get('Description', 'N/A')}"
        ]
        # Add aggregated data
        for col in combined_cols:
            prefix = col.replace('_Combined', '').replace('_', ' ') # Make prefix readable
            doc_text_parts.append(f"{prefix}: {row.get(col, 'N/A')}")

        doc_text = "\n".join(doc_text_parts)
        documents.append(doc_text)
        doc_metadata.append({
            'title': row.get('Title', 'N/A'),
            'code': row.get('O*NET-SOC Code', 'N/A'),
            'original_text': doc_text
        })
    logging.info(f"Created {len(documents)} text documents for embedding.")
    return documents, doc_metadata

# --- 3. Embed Documents --- (Same as before, slightly improved logging)

def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=100):
    """Embeds a list of texts using the Gemini embedding model with batching and retries."""
    embeddings = []
    total_texts = len(texts)
    logging.info(f"Starting embedding process for {total_texts} documents...")
    start_time = time.time()

    for i in range(0, total_texts, batch_size):
        batch = texts[i:min(i + batch_size, total_texts)]
        batch_num = (i // batch_size) + 1
        total_batches = (total_texts + batch_size - 1) // batch_size
        logging.info(f"  Embedding batch {batch_num}/{total_batches} (size {len(batch)})...")

        retries = 2
        for attempt in range(retries + 1):
            try:
                result = genai.embed_content(model=EMBEDDING_MODEL,
                                             content=batch,
                                             task_type=task_type)
                # Ensure the result is a list of embeddings
                batch_embeddings = result.get('embedding', [])
                if isinstance(batch_embeddings, list) and len(batch_embeddings) == len(batch):
                     embeddings.extend(batch_embeddings)
                     # Add a small delay to respect potential rate limits
                     time.sleep(0.5 + attempt * 0.5) # Increase delay slightly on retry success
                     break # Success, exit retry loop
                else:
                     raise ValueError(f"Unexpected embedding result structure or length mismatch: {result}")

            except Exception as e:
                logging.warning(f"    Attempt {attempt + 1}/{retries + 1}: Error embedding batch {batch_num}: {e}")
                if attempt < retries:
                    wait_time = 2 ** attempt # Exponential backoff
                    logging.info(f"    Retrying batch {batch_num} in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"    Failed embedding batch {batch_num} after {retries + 1} attempts.")
                    # Add None placeholders for the failed batch
                    embeddings.extend([None] * len(batch))

    end_time = time.time()
    # Filter out None values where embedding failed
    valid_embeddings_with_indices = [(i, emb) for i, emb in enumerate(embeddings) if emb is not None]
    num_failed = total_texts - len(valid_embeddings_with_indices)

    logging.info(f"Embedding finished in {end_time - start_time:.2f} seconds.")
    if num_failed > 0:
         logging.warning(f"{num_failed} out of {total_texts} documents failed to embed.")
    else:
         logging.info(f"Successfully embedded all {total_texts} documents.")

    # Return only the embeddings and the indices they correspond to in the original list
    original_indices = [item[0] for item in valid_embeddings_with_indices]
    valid_embeddings = [item[1] for item in valid_embeddings_with_indices]

    return valid_embeddings, original_indices


# --- 4. Build Vector Store (FAISS) --- (Same as before)
def build_faiss_index(embeddings):
    """Builds a FAISS index for efficient similarity search."""
    if not embeddings:
        logging.error("No valid embeddings provided to build index.")
        return None
    try:
        vector_dim = len(embeddings[0]) # Get dimension from first embedding
        embeddings_np = np.array(embeddings).astype('float32')
        if embeddings_np.ndim != 2:
             raise ValueError(f"Embeddings numpy array must be 2D, but got shape {embeddings_np.shape}")

        index = faiss.IndexFlatL2(vector_dim)
        index.add(embeddings_np) # Add the vectors to the index
        logging.info(f"FAISS index built successfully with {index.ntotal} vectors of dimension {index.d}.")
        return index
    except Exception as e:
        logging.error(f"Error building FAISS index: {e}")
        return None


# --- 5. Retrieval --- (Updated to use filtered metadata)
def find_similar_documents(query, index, query_embedding_model, doc_metadata_filtered, top_k=3):
    """Finds top_k most similar documents to the query using the filtered metadata."""
    if index is None:
        logging.error("FAISS index not available for search.")
        return []
    if not doc_metadata_filtered:
         logging.warning("Document metadata is empty, cannot retrieve.")
         return []

    try:
        # Embed the user query
        query_embedding_result = genai.embed_content(model=query_embedding_model,
                                                    content=[query], # Embed as a list
                                                    task_type="RETRIEVAL_QUERY")
        query_vector = np.array(query_embedding_result['embedding']).astype('float32')

        # Search the FAISS index
        distances, indices = index.search(query_vector, top_k)

        # Retrieve the corresponding original documents using the filtered metadata
        # indices[0] contains the indices *within the FAISS index*, which correspond
        # directly to the order in doc_metadata_filtered
        retrieved_docs = []
        for i in indices[0]:
             if 0 <= i < len(doc_metadata_filtered):
                 retrieved_docs.append(doc_metadata_filtered[i]['original_text'])
             else:
                 logging.warning(f"FAISS returned index {i} which is out of bounds for filtered metadata (size {len(doc_metadata_filtered)}).")

        logging.info(f"Retrieved {len(retrieved_docs)} documents for query: '{query[:50]}...'")
        return retrieved_docs

    except Exception as e:
        logging.error(f"Error during similarity search for query '{query[:50]}...': {e}")
        return []

# --- 6. Generation --- (Same as before, slightly better error message)
def generate_answer(query, context_docs, generation_model_name):
    """Generates an answer using Gemini, grounding it in the retrieved context."""
    if not context_docs:
        return "I couldn't find relevant information in the O*NET data to answer that specific question."

    context_str = "\n\n---\n\n".join(context_docs)
    prompt = f"""You are a helpful chatbot knowledgeable about careers based on the O*NET database.
Answer the following user query based *only* on the provided context information from O*NET.
Do not add information not present in the context. If the context doesn't contain the specific answer, clearly state that the provided details don't include that information, but you can summarize what is available.

**Context from O*NET:**
{context_str}

**User Query:**
{query}

**Answer:**
"""
    logging.info(f"Generating answer for query: '{query[:50]}...'")
    try:
        model = genai.GenerativeModel(generation_model_name)
        # Configure safety settings to be less restrictive if needed, but be careful
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]
        # response = model.generate_content(prompt, safety_settings=safety_settings)

        response = model.generate_content(prompt)

        # Check for blocked response due to safety or other reasons
        if not response.parts:
             logging.warning(f"Generation response blocked. Prompt Feedback: {response.prompt_feedback}")
             # Provide a more informative message to the user
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             return f"Sorry, I couldn't generate a response for that query. The content may have been blocked (Reason: {block_reason}). Please try rephrasing your question."

        return response.text
    except Exception as e:
        logging.error(f"Error during generation with model {generation_model_name}: {e}")
        # You might want to check the type of exception here for more specific messages
        return "Sorry, I encountered an error while trying to generate a response. Please try again."


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting O*NET Chatbot ---")
    logging.info(f"Loading O*NET data from: {ONET_DATA_DIR}")

    merged_onet_data = load_and_merge_onet_data(ONET_DATA_DIR, ONET_FILES_CONFIG)

    if merged_onet_data is not None and not merged_onet_data.empty:
        logging.info("Creating document chunks from merged data...")
        all_documents, all_doc_metadata = create_documents(merged_onet_data)

        logging.info("Embedding documents...")
        # Embed all documents
        doc_embeddings, valid_original_indices = embed_texts(all_documents)

        if doc_embeddings: # Proceed only if some embeddings were successful
            # Filter the metadata to match the successful embeddings
            # valid_original_indices contains the original indices (0 to N-1) of the documents that were successfully embedded.
            # We need to create a new metadata list containing only the items at these indices.
            filtered_doc_metadata = [all_doc_metadata[i] for i in valid_original_indices]

            logging.info("Building FAISS index...")
            faiss_index = build_faiss_index(doc_embeddings)

            if faiss_index:
                logging.info("\n--- O*NET Chatbot Ready ---")
                print("Ask questions about occupations (e.g., 'Tell me about Software Developers', 'What skills do Nurses need?', 'What tools do plumbers use?'). Type 'quit' to exit.")

                while True:
                    user_query = input("\nYou: ")
                    if user_query.lower() == 'quit':
                        break
                    if not user_query.strip():
                        continue

                    print("Bot: Thinking...")
                    # 1. Retrieve relevant documents using filtered metadata
                    retrieved_context = find_similar_documents(
                        user_query,
                        faiss_index,
                        EMBEDDING_MODEL, # Pass the model name used for query embedding
                        filtered_doc_metadata, # Pass the filtered metadata
                        top_k=4 # Retrieve slightly more context
                    )

                    # 2. Generate answer based on context
                    answer = generate_answer(user_query, retrieved_context, GENERATION_MODEL)
                    print(f"Bot: {answer}")
            else:
                logging.error("Failed to build the search index. Chatbot cannot start.")
        else:
            logging.error("No documents were successfully embedded. Chatbot cannot start.")
    else:
        logging.error("Failed to load or merge O*NET data. Exiting.")

    logging.info("--- O*NET Chatbot Shutting Down ---")
