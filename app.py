import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Model Loading ---

# Use st.cache_resource to load the model only once and cache it for subsequent runs.
# This is crucial for performance as downloading and loading the model can be slow.
@st.cache_resource
def load_grounding_model():
    """
    Loads and returns the HuggingFace text-classification pipeline.
    """
    st.info("Downloading the classification model... This may take a moment on the first run.")
    # Initialize the pipeline with the specified model
    classifier = pipeline(
        "text-classification",
        model="dejanseo/query-grounding",
        top_k=None # Ensures we get scores for both labels
    )
    return classifier

# --- Main Application UI ---

st.set_page_config(page_title="Query Grounding Analyzer", layout="wide")

st.title("🔎 Query Grounding Analyzer")
st.markdown(
    "This tool uses the `dejanseo/query-grounding` model from HuggingFace to "
    "determine the probability that a search query is **grounded** (seeking factual, objective information)."
)

# --- User Input ---
keywords_input = st.text_area(
    "Enter keywords or search queries, one per line:",
    height=200,
    placeholder="what is the capital of france\nbest travel destinations\nhow to bake a cake\nmeaning of life"
)

analyze_button = st.button("📊 Analyze Queries")

# --- Analysis and Results ---
if analyze_button and keywords_input:
    # 1. Load the model from cache
    classifier = load_grounding_model()
    
    # 2. Process the input text into a list of queries
    keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
    
    results = []
    
    # 3. Show a spinner while processing
    with st.spinner(f"Analyzing {len(keywords)} queries..."):
        # 4. Run each keyword through the model
        model_outputs = classifier(keywords)
        
        # 5. Parse the results to find the grounding score
        for keyword, output in zip(keywords, model_outputs):
            grounding_score = 0.0 # Default score
            for label in output:
                if label['label'] == 'grounded':
                    grounding_score = label['score']
                    break # Stop once we find the 'grounded' label
            
            results.append({
                "Query": keyword,
                "Grounding Chance": grounding_score
            })

    st.success("Analysis complete!")
    
    # 6. Display the results in a DataFrame
    df = pd.DataFrame(results)
    
    # Format the 'Grounding Chance' column as a percentage
    st.dataframe(
        df,
        column_config={
            "Query": st.column_config.TextColumn("Query", width="large"),
            "Grounding Chance": st.column_config.ProgressColumn(
                "Grounding Chance (%)",
                help="The model's confidence that the query is seeking factual, objective information.",
                format="%.1f%%",
                min_value=0,
                max_value=1,
            ),
        },
        use_container_width=True,
        hide_index=True
    )

elif analyze_button:
    st.warning("Please enter at least one keyword to analyze.")
