import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# 1. Setup Page Layout & Custom CSS
st.set_page_config(page_title="GovProcure AI", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .entity-tag {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .org-tag { background-color: #4B0082; color: white; }
    .loc-tag { background-color: #006400; color: white; }
    .prod-tag { background-color: #8B0000; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Federal Procurement Intelligence")
st.markdown("Analyze spending trends and search contracts using AI semantic matching & entity extraction.")
st.divider()

# 2. Caching models and data
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_ner_model():
    return spacy.load("en_core_web_sm")

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_contracts.csv")
    df['Cleaned_Description'] = df['Cleaned_Description'].fillna("")
    embeddings = np.load("data/contract_embeddings.npy")
    return df, embeddings

semantic_model = load_semantic_model()
nlp = load_ner_model()
df, embeddings = load_data()

# Helper function for NER
def extract_entities_ui(text):
    if not text: return {}
    doc = nlp(text)
    return {
        "ORG": list({ent.text for ent in doc.ents if ent.label_ == "ORG"}),
        "LOC": list({ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}),
        "PROD": list({ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "FAC"]})
    }

# 3. Sidebar Configuration
with st.sidebar:
    st.header("🎯 Dashboard Filters")
    min_amount = st.number_input("Minimum Award Amount ($)", min_value=0, value=100000)
    agencies = ["All"] + sorted(df['Awarding Agency'].dropna().unique().tolist())
    selected_agency = st.selectbox("Select Agency", agencies)

filtered_df = df[df["Award Amount"] >= min_amount]
if selected_agency != "All":
    filtered_df = filtered_df[filtered_df["Awarding Agency"] == selected_agency]

# 4. Top KPI Metrics Dashboard
st.subheader("📊 High-Level Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Contracts", f"{len(filtered_df):,}")
col2.metric("Total Obligated Spend", f"${filtered_df['Award Amount'].sum():,.0f}")
col3.metric("Average Award Size", f"${filtered_df['Award Amount'].mean():,.0f}")
col4.metric("Unique Vendors", f"{filtered_df['Recipient Name'].nunique():,}")

st.write("") 

# 5. Interactive Data Visualizations
st.subheader("📈 Spending Analytics")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    agency_spend = filtered_df.groupby("Awarding Agency")["Award Amount"].sum().reset_index()
    agency_spend = agency_spend.sort_values(by="Award Amount", ascending=False).head(10)
    fig_bar = px.bar(agency_spend, x="Award Amount", y="Awarding Agency", orientation='h',
                     title="Top Agencies by Total Spend", color="Award Amount", color_continuous_scale="Blues")
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    fig_scatter = px.scatter(filtered_df.head(1000), x="Awarding Agency", y="Award Amount", 
                             color="Awarding Agency", size="Award Amount", hover_name="Recipient Name",
                             title="Distribution of Top 1000 Contract Awards")
    fig_scatter.update_xaxes(showticklabels=False)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# 6. AI Semantic Search Section with NER
st.subheader("🧠 AI Semantic Search Engine")
query = st.text_input("Describe the contract you are looking for in plain English:", 
                      placeholder="e.g., Cybersecurity vulnerability assessment and defense...")

if st.button("Search Contracts", type="primary") and query:
    with st.spinner("Neural network analyzing contract semantic context..."):
        query_vector = semantic_model.encode([query])
        scores = cosine_similarity(query_vector, embeddings)[0]
        
        df_results = filtered_df.copy()
        df_results["Match Score"] = [scores[i] for i in df_results.index] 
        top_matches = df_results.sort_values(by="Match Score", ascending=False).head(10)
        
        st.success(f"Top matches found for: '{query}'")
        
        for index, row in top_matches.iterrows():
            with st.expander(f"⭐ Score: {row['Match Score']:.3f} | {row['Recipient Name']} | ${row['Award Amount']:,.2f}"):
                st.markdown(f"**Agency:** {row['Awarding Agency']}")
                st.markdown(f"**Description:** {row['Cleaned_Description']}")
                
                # Perform NER dynamically on the top results
                entities = extract_entities_ui(row['Cleaned_Description'])
                
                tags_html = ""
                for org in entities.get("ORG", []):
                    tags_html += f'<span class="entity-tag org-tag">🏢 {org}</span>'
                for loc in entities.get("LOC", []):
                    tags_html += f'<span class="entity-tag loc-tag">📍 {loc}</span>'
                for prod in entities.get("PROD", []):
                    tags_html += f'<span class="entity-tag prod-tag">⚙️ {prod}</span>'
                
                if tags_html:
                    st.markdown("**Extracted Entities:**")
                    st.markdown(tags_html, unsafe_allow_html=True)