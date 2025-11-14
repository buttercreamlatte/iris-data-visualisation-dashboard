# streamlit_iris_app.py (restored version)
import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly.express as px

st.set_page_config(page_title="Iris Data Exploration", layout="wide", initial_sidebar_state="expanded")

# Load data
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# tidy column names
df.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in df.columns]

# Sidebar
st.sidebar.header("Data Selection")

# Species select (multi-select)
species_options = list(df['species'].cat.categories)
selected_species = st.sidebar.multiselect("Select species", options=species_options, default=species_options)

# Sepal length slider
min_sepal = float(df.sepal_length.min())
max_sepal = float(df.sepal_length.max())
sepal_range = st.sidebar.slider("Sepal length range (cm)", min_value=min_sepal, max_value=max_sepal, value=(min_sepal, max_sepal))

# Toggle to show raw table
show_raw = st.sidebar.checkbox("Show raw data table", value=False)

#Apply filters
filtered = df[
    (df['species'].isin(selected_species)) &
    (df.sepal_length >= sepal_range[0]) &
    (df.sepal_length <= sepal_range[1])
]

# Theme
color_map = {"setosa": "#003f5c", "versicolor": "#d45087", "virginica": "#2f4b7c"}

# Layout
st.title("Iris Data Visualisation Dashboard")
st.markdown("Let's dive into the Iris dataset with interactive filtering, clean visuals, and insightful summary statistics.")

# Top metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Filtered Rows", len(filtered))
col2.metric("Average Sepal Length (cm)", f"{filtered.sepal_length.mean():.2f}")
col3.metric("Average Petal Length (cm)", f"{filtered.petal_length.mean():.2f}")

st.markdown("---")

# Visualisations
vis_expander = st.expander("Visualisations", expanded=True)
with vis_expander:
    st.subheader("Scatter Plot: Sepal vs Petal")
    with st.expander("About this scatter plot"):
        st.write("This visual compares sepal length and petal length across different Iris species.")
    fig_scatter = px.scatter(
        filtered,
        x='sepal_length',
        y='petal_length',
        color='species',
        color_discrete_map=color_map,
        labels={"sepal_length": "Sepal Length (cm)", "petal_length": "Petal Length (cm)"},
        title='Sepal Length vs Petal Length'
    )
    fig_scatter.update_layout(
        legend_title_text='Species',
        template='plotly_white',
        paper_bgcolor='rgba(255,255,255,0.2)',
        plot_bgcolor='rgba(255,255,255,0.2)'
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_chart")

    st.subheader("Histogram: Sepal Length Distribution")
    with st.expander("About this histogram"):
        st.write("This histogram shows how sepal length values are distributed within the filtered dataset.")
    fig_hist = px.histogram(
        filtered,
        x='sepal_length',
        nbins=20,
        color='species',
        barmode='overlay',
        color_discrete_map=color_map,
        labels={'sepal_length': 'Sepal Length (cm)'},
        title='Distribution of Sepal Length by Species'
    )
    fig_hist.update_traces(opacity=0.6)
    fig_hist.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(255,255,255,0.2)',
        plot_bgcolor='rgba(255,255,255,0.2)'
    )
    st.plotly_chart(fig_hist, use_container_width=True, key="hist_chart")

data_expander = st.expander("Data Summary", expanded=False)
with data_expander:
    st.subheader("Summary of Data")
    st.write(filtered.describe())

    st.write("**Species Counts**")
    st.table(filtered['species'].value_counts().rename_axis('species').reset_index(name='count'))

    if show_raw:
        st.subheader("Filtered Raw Data")
        st.dataframe(filtered.reset_index(drop=True))

# Footer
st.markdown("---")
st.caption("Najwa Mahmood | Bachelor of Computer Science")

# Small CSS tweak
st.markdown(
    """
    <style>
html, body, .stApp {
    margin: 0 !important;
    padding: 0 !important;
}

.stApp {
    background: linear-gradient(135deg, #f6e6f7 0%, #e2f0ff 50%, #f9f5ff 100%) !important;
}

header[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
    height: 0rem !important;
}

[data-testid="stSidebar"] {
    background: #faf4ff !important;
    border-right: 1px solid #e6d9f7;
}

.stApp .css-1d391kg {padding-top: 0rem !important;}
    </style>
    """,
    unsafe_allow_html=True
)
