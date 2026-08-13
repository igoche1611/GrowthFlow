import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="GrowthFlow Retention Dashboard", layout="wide")
st.title("GrowthFlow Customer Retention Dashboard")
st.caption("Cohort retention, churn, customer segmentation, revenue and engagement analysis")

@st.cache_data
def load_data():
    master = pd.read_csv("growthflow_master_dataset.csv")
    segments = pd.read_csv("growthflow_customer_segments.csv")
    retention = pd.read_csv("growthflow_cohort_retention_matrix.csv", index_col=0)
    churn = pd.read_csv("growthflow_churn_matrix.csv", index_col=0)
    milestones = pd.read_csv("growthflow_churn_milestones.csv", index_col=0)
    retention.columns = [int(c) if str(c).isdigit() else c for c in retention.columns]
    churn.columns = [int(c) if str(c).isdigit() else c for c in churn.columns]
    return master, segments, retention, churn, milestones

master, segments, retention, churn, milestones = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Retention & Churn", "Customer Segments", "Revenue & Engagement"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{master['customer_id'].nunique():,}")
    c2.metric("Customer-Month Records", f"{len(master):,}")
    if "monthly_revenue" in master.columns:
        c3.metric("Average Monthly Revenue", f"{master['monthly_revenue'].mean():,.2f}")
    if "is_active" in master.columns:
        c4.metric("Active Record Rate", f"{master['is_active'].mean()*100:.1f}%")
    st.subheader("Analytical Dataset")
    st.dataframe(master.head(100), use_container_width=True)

with tab2:
    st.subheader("Average Retention Over Time")
    avg_retention = retention.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_retention.index, avg_retention.values, marker="o")
    ax.set_xlabel("Months Since Subscription")
    ax.set_ylabel("Retention Rate (%)")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Average Churn Over Time")
    avg_churn = churn.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_churn.index, avg_churn.values, marker="o")
    ax.set_xlabel("Months Since Subscription")
    ax.set_ylabel("Churn Rate (%)")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Churn at Key Milestones")
    st.dataframe(milestones.round(2), use_container_width=True)

with tab3:
    st.subheader("Customer Segment Profiles")
    segment_col = "cluster" if "cluster" in segments.columns else (
        "segment" if "segment" in segments.columns else None
    )
    if segment_col:
        st.bar_chart(segments[segment_col].value_counts().sort_index())
        numeric = segments.select_dtypes(include=np.number).columns.tolist()
        profile_cols = [c for c in numeric if c != segment_col]
        profile = segments.groupby(segment_col)[profile_cols].mean().round(2)
        st.dataframe(profile, use_container_width=True)
    else:
        st.warning("No cluster/segment column was found.")
    st.dataframe(segments.head(100), use_container_width=True)

with tab4:
    left, right = st.columns(2)
    with left:
        st.subheader("Revenue")
        if "monthly_revenue" in master.columns and "activity_month" in master.columns:
            rev = master.groupby("activity_month")["monthly_revenue"].mean()
            st.line_chart(rev)
        else:
            st.info("Revenue fields are not available.")
    with right:
        st.subheader("Engagement")
        if "response_rate" in master.columns and "activity_month" in master.columns:
            eng = master.groupby("activity_month")["response_rate"].mean()
            st.line_chart(eng)
        elif "engagement_count" in master.columns and "activity_month" in master.columns:
            eng = master.groupby("activity_month")["engagement_count"].mean()
            st.line_chart(eng)
        else:
            st.info("Engagement fields are not available.")

st.divider()
st.caption("GrowthFlow Customer Retention Strategy Optimization")
