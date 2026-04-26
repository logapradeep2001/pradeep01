import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("AI Conversational Data Analytics System")

# ---------------------------
# SESSION STATE (MEMORY)
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# FILE UPLOAD
# ---------------------------
file = st.file_uploader("Upload CSV Dataset")

if file:
    df = pd.read_csv(file, dtype=str, low_memory=False)
    st.session_state.df = df

# ---------------------------
# CHAT INPUT
# ---------------------------
query = st.text_input("Ask anything about your data")

if st.session_state.df is not None:

    df = st.session_state.df
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    if query:

        q = query.lower()
        st.session_state.history.append(query)

        # ---------------------------
        # SHOW COLUMNS
        # ---------------------------
        if "column" in q:
            st.write("Columns:", df.columns.tolist())

        # ---------------------------
        # CLEAN DATA
        # ---------------------------
        elif "clean" in q or "missing" in q:
            for col in numeric_df.columns:
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

            st.session_state.df = numeric_df
            st.write("Data cleaned successfully")

        # ---------------------------
        # BAR CHART
        # ---------------------------
        elif "bar" in q:
            col1 = df.columns[0]
            col2 = df.columns[1]

            try:
                chart = df.groupby(col1)[col2].count()
                st.bar_chart(chart)
                st.write(f"Bar chart of {col1} vs {col2}")
            except:
                st.write("Cannot generate bar chart")

        # ---------------------------
        # SCATTER PLOT
        # ---------------------------
        elif "scatter" in q:

            cols = numeric_df.dropna(axis=1, how='all').columns

            if len(cols) >= 2:
                x = cols[0]
                y = cols[1]

                fig, ax = plt.subplots()
                ax.scatter(numeric_df[x], numeric_df[y])
                ax.set_xlabel(x)
                ax.set_ylabel(y)

                st.pyplot(fig)
                st.write(f"Scatter plot: {x} vs {y}")

        # ---------------------------
        # LINE / TREND
        # ---------------------------
        elif "trend" in q or "line" in q:
            st.line_chart(numeric_df)
            st.write("Trend visualization shown")

        # ---------------------------
        # SUMMARY
        # ---------------------------
        elif "summary" in q or "describe" in q:
            st.text(numeric_df.describe().to_string())

        # ---------------------------
        # FILTER
        # ---------------------------
        elif "top" in q:
            st.write(df.head(5))

        # ---------------------------
        # SEGMENTATION
        # ---------------------------
        elif "segment" in q or "cluster" in q:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=3, n_init=10)
            clusters = kmeans.fit_predict(numeric_df.fillna(0))

            st.write("Cluster distribution:", np.bincount(clusters))

        # ---------------------------
        # INSIGHTS
        # ---------------------------
        elif "insight" in q or "analysis" in q:
            st.write("• Data shows variation across variables")
            st.write("• Some segments may be underperforming")
            st.write("• Opportunities exist for optimization")

        # ---------------------------
        # MODIFY / RE-RUN
        # ---------------------------
        elif "modify" in q or "change" in q:
            st.write("You can refine your request like:")
            st.write("- show top 10")
            st.write("- filter by column")
            st.write("- compare variables")


        else:
            st.write("Try asking: show columns, clean data, bar chart, scatter plot, insights")


st.subheader("Chat History")
for h in st.session_state.history:
    st.write("-", h)