import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Iris Data Exploration",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in df.columns]

# Feature Mapping
feature_name_map = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_length": "Petal Length (cm)",
    "petal_width": "Petal Width (cm)"
}

readable_features = list(feature_name_map.values())
readable_to_original = {v: k for k, v in feature_name_map.items()}

# Theme
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5f0fa 0%, #ffffff 60%);
        color: #4b1f70;
    }
    section[data-testid="stSidebar"] {
        background-color: #f3e6fa;
        border-right: 1px solid #e0c7f0;
    }
    h1, h2, h3 {
        color: #4b1f70;
    }
    .plot-card {
        background-color: rgba(255,255,255,0.85);
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(75,31,112,0.1);
        margin-bottom: 12px;
    }
    header[data-testid="stHeader"] {
        background-color: transparent;
        height: 0;
    }
    
    /* Sidebar Header Purple */
    section[data-testid="stSidebar"] h2 {
        color: #4b1f70 !important;
    }
    
        /* Selected tag colour in multiselect → purple */
    span[data-baseweb="tag"] {
        background-color: #d9b3ff !important;
        color: #4b1f70 !important;
        border: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Data Selection")

species_options = list(df['species'].cat.categories)
selected_species = st.sidebar.multiselect(
    "Select Species", options=species_options, default=species_options
)

x_feature_readable = st.sidebar.selectbox("X-Axis Feature", options=readable_features, index=0)
y_feature_readable = st.sidebar.selectbox("Y-Axis Feature", options=readable_features, index=2)

x_feature = readable_to_original[x_feature_readable]
y_feature = readable_to_original[y_feature_readable]

show_raw = st.sidebar.checkbox("Show raw data", value=False)

filtered = df[df['species'].isin(selected_species)]

# Page Header
st.title("Iris Dataset Exploration Dashboard")
st.write("Interactively explore the Iris dataset with multiple visualisations and summary statistics.")

m1, m2, m3 = st.columns(3)
m1.metric("Filtered Rows", len(filtered))
m2.metric("Average Sepal Length", f"{filtered.sepal_length.mean():.2f} cm")
m3.metric("Average Petal Length", f"{filtered.petal_length.mean():.2f} cm")

st.markdown("---")

# Scattered Plot
with st.expander("Scatter Plot", expanded=True):
    st.write("Scatter plot of selected features to observe relationships between sepal and petal measurements.")
    fig_scatter = px.scatter(
        filtered,
        x=x_feature,
        y=y_feature,
        color="species",
        labels={**feature_name_map, "species": "Species"},
        color_discrete_map={"setosa": "#d896ff", "versicolor": "#be29ec", "virginica": "#800080"},
        title=f"{feature_name_map[x_feature]} vs {feature_name_map[y_feature]}"
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_iris")

# Histogram
with st.expander("Histogram", expanded=False):
    st.write("Histogram showing the distribution of the selected X-axis feature by species.")
    fig_hist = px.histogram(
        filtered,
        x=x_feature,
        color="species",
        barmode="overlay",
        labels={**feature_name_map, "species": "Species"},
        color_discrete_map={"setosa": "#d896ff", "versicolor": "#be29ec", "virginica": "#800080"},
        title=f"Distribution of {feature_name_map[x_feature]} by Species"
    )
    fig_hist.update_traces(opacity=1)
    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_hist, use_container_width=True, key="hist_iris")

# Box Plot
with st.expander("Box Plot", expanded=False):
    st.write("Box plot for selected X-axis feature across species to visualise spread, quartiles, and outliers.")
    fig_box = px.box(
        filtered,
        x="species",
        y=x_feature,
        color="species",
        labels={**feature_name_map, "species": "Species"},
        color_discrete_map={"setosa": "#d896ff", "versicolor": "#be29ec", "virginica": "#800080"},
        title=f"{feature_name_map[x_feature]} by Species (Box Plot)"
    )
    fig_box.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_box, use_container_width=True, key="box_iris")

# Pairwise Scatter Matrix
with st.expander("Pairwise Scatter Matrix", expanded=False):
    st.write("Pairwise scatter plots for all features to quickly observe inter-feature relationships.")
    pair_df = filtered.copy()
    pair_df = pair_df.rename(columns=feature_name_map)
    fig_pair = px.scatter_matrix(
        pair_df,
        dimensions=list(feature_name_map.values()),
        color="species",
        color_discrete_map={"setosa": "#d896ff", "versicolor": "#be29ec", "virginica": "#800080"},
        title="Pairwise Feature Relationships"
    )
    fig_pair.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pair, use_container_width=True, key="pair_iris")

# Data Summary
with st.expander("Data Summary", expanded=False):
    st.subheader("Descriptive Statistics")
    st.dataframe(filtered.describe().round(2))
    st.markdown("### Counts by Species")
    st.table(filtered['species'].value_counts().rename_axis("Species").reset_index(name="Count"))

    if show_raw:
        st.markdown("### Raw Data")
        st.dataframe(filtered.reset_index(drop=True))

# Footer
st.markdown("---")
st.caption("Najwa Mahmood | Bachelor of Computer Science")
