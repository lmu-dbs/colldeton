import streamlit as st

st.set_page_config(page_title="Collusion Detection Demo")

st.sidebar.success("Select a method above.")

st.write("## Collusion Detection Demo")
st.markdown("""
- **HAC (Hierarchical Agglomerative Clustering):**
  - A clustering technique that groups similar objects based on their attributes using a hierarchical structure.
  - Utilizes dendrogram visualization to represent the clustering process and identify colluding entities.
  - Offers flexibility in selecting linkage criteria and thresholds for optimal clustering results.

- **Z-Scoring:**
  - A statistical method that standardizes data points by converting them into z-scores, reflecting how many standard deviations a point is from the mean.
  - Applied to detect outliers or anomalies in relational data, helping to identify potential collusions.
  - Visualized through treemaps and bar charts for intuitive understanding of group sizes and z-score distributions.
""")

