import streamlit as st
import pandas as pd
from transformers import pipeline

# --- Model Loading ---

@st.cache_resource
def load_grounding_model():
    """
    Loads and returns the HuggingFace text-classification pipeline.
    """
    st.info("Downloading the classification model... This may take a moment on the first run.")
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
    classifier = load_grounding_model()
    keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
    results = []

    with st.spinner(f"Analyzing {len(keywords)} queries..."):
        model_outputs = classifier(keywords)

        for keyword, output in zip(keywords, model_outputs):
            grounding_score = 0.0
            for label in output:
                # --- THIS IS THE CORRECTED LINE ---
                # Check for either the text label or the raw internal label.
                if label['label'] == 'grounded' or label['label'] == 'LABEL_1':
                    grounding_score = label['score']
                    break

            results.append({
                "Query": keyword,
                "Grounding Chance": grounding_score
            })

    st.success("Analysis complete!")

    df = pd.DataFrame(results)

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
