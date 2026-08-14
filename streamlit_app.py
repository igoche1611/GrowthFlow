from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GrowthFlow Retention Dashboard",
    layout="wide",
)

st.title("GrowthFlow Customer Retention Dashboard")
st.caption(
    "Cohort retention, churn, customer segmentation, revenue and engagement analysis"
)


# --------------------------------------------------
# DATA LOCATION
# --------------------------------------------------

# Your GrowthFlow CSV files are stored in:
# C:\Users\<your-user-name>\
# Path.home() resolves that location automatically.
DATA_DIR = Path.home()

MASTER_FILE = DATA_DIR / "growthflow_master_dataset.csv"
SEGMENTS_FILE = DATA_DIR / "growthflow_customer_segments.csv"
RETENTION_FILE = DATA_DIR / "growthflow_cohort_retention_matrix.csv"
CHURN_FILE = DATA_DIR / "growthflow_churn_matrix.csv"
MILESTONES_FILE = DATA_DIR / "growthflow_churn_milestones.csv"

REQUIRED_FILES = [
    MASTER_FILE,
    SEGMENTS_FILE,
    RETENTION_FILE,
    CHURN_FILE,
    MILESTONES_FILE,
]

missing_files = [str(path) for path in REQUIRED_FILES if not path.exists()]

if missing_files:
    st.error(
        "The dashboard cannot start because these required files were not found:\n\n"
        + "\n".join(f"- {path}" for path in missing_files)
    )
    st.stop()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

# Passing file modification times into the cached function ensures Streamlit
# reloads the data automatically whenever any source CSV changes.
file_versions = tuple(path.stat().st_mtime_ns for path in REQUIRED_FILES)


@st.cache_data
def load_data(file_versions):
    # file_versions is intentionally passed so the cache invalidates when a CSV changes.
    _ = file_versions

    master = pd.read_csv(MASTER_FILE)
    segments = pd.read_csv(SEGMENTS_FILE)
    retention = pd.read_csv(RETENTION_FILE, index_col=0)
    churn = pd.read_csv(CHURN_FILE, index_col=0)
    milestones = pd.read_csv(MILESTONES_FILE, index_col=0)

    # Convert master date columns to proper datetimes.
    for column in ["activity_month", "subscription_date", "cohort"]:
        if column in master.columns:
            master[column] = pd.to_datetime(master[column], errors="coerce")

    # Convert cohort-matrix month labels from strings such as "12" to integer 12.
    retention.columns = [
        int(column) if str(column).isdigit() else column
        for column in retention.columns
    ]

    churn.columns = [
        int(column) if str(column).isdigit() else column
        for column in churn.columns
    ]

    return master, segments, retention, churn, milestones


master, segments, retention, churn, milestones = load_data(file_versions)


# --------------------------------------------------
# SHARED VALUES
# --------------------------------------------------

customer_count = master["customer_id"].nunique()
record_count = len(master)

avg_monthly_revenue = (
    master["monthly_revenue"].mean()
    if "monthly_revenue" in master.columns
    else np.nan
)

avg_12_month_retention = (
    retention[12].mean()
    if 12 in retention.columns
    else np.nan
)


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview",
        "Retention & Churn",
        "Customer Segments",
        "Revenue & Engagement",
    ]
)


# ==================================================
# TAB 1 — OVERVIEW
# ==================================================

with tab1:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        f"{customer_count:,}",
    )

    c2.metric(
        "Customer-Month Records",
        f"{record_count:,}",
    )

    c3.metric(
        "Average Monthly Revenue",
        f"{avg_monthly_revenue:,.2f}"
        if pd.notna(avg_monthly_revenue)
        else "N/A",
    )

    c4.metric(
        "Avg. 12-Month Retention",
        f"{avg_12_month_retention:.1f}%"
        if pd.notna(avg_12_month_retention)
        else "N/A",
    )

    st.subheader("Analytical Dataset")

    overview_data = master.head(100).copy()

    # Format dates for display only; the original master DataFrame remains datetime.
    for column in ["activity_month", "subscription_date", "cohort"]:
        if column in overview_data.columns:
            overview_data[column] = overview_data[column].dt.strftime("%Y-%m-%d")

    st.dataframe(
        overview_data,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# TAB 2 — RETENTION & CHURN
# ==================================================

with tab2:
    avg_retention = retention.mean(axis=0)
    avg_churn = churn.mean(axis=0)

    month_3_churn = (
        milestones["month_3_churn"].mean()
        if "month_3_churn" in milestones.columns
        else np.nan
    )

    month_6_churn = (
        milestones["month_6_churn"].mean()
        if "month_6_churn" in milestones.columns
        else np.nan
    )

    month_12_churn = (
        milestones["month_12_churn"].mean()
        if "month_12_churn" in milestones.columns
        else np.nan
    )

    month_12_retention = (
        retention[12].mean()
        if 12 in retention.columns
        else np.nan
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Avg. 3-Month Churn",
        f"{month_3_churn:.1f}%"
        if pd.notna(month_3_churn)
        else "N/A",
    )

    m2.metric(
        "Avg. 6-Month Churn",
        f"{month_6_churn:.1f}%"
        if pd.notna(month_6_churn)
        else "N/A",
    )

    m3.metric(
        "Avg. 12-Month Churn",
        f"{month_12_churn:.1f}%"
        if pd.notna(month_12_churn)
        else "N/A",
    )

    m4.metric(
        "Avg. 12-Month Retention",
        f"{month_12_retention:.1f}%"
        if pd.notna(month_12_retention)
        else "N/A",
    )

    retention_chart_data = pd.DataFrame(
        {
            "Month": pd.to_numeric(avg_retention.index, errors="coerce"),
            "Retention Rate": avg_retention.values,
        }
    ).dropna(subset=["Month"])

    churn_chart_data = pd.DataFrame(
        {
            "Month": pd.to_numeric(avg_churn.index, errors="coerce"),
            "Churn Rate": avg_churn.values,
        }
    ).dropna(subset=["Month"])

    left, right = st.columns(2)

    with left:
        st.subheader("Average Cohort Retention")

        retention_chart = (
            alt.Chart(retention_chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Month:Q",
                    title="Months Since Subscription",
                    axis=alt.Axis(tickMinStep=1),
                ),
                y=alt.Y(
                    "Retention Rate:Q",
                    title="Retention Rate (%)",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip("Month:Q", title="Month", format=".0f"),
                    alt.Tooltip(
                        "Retention Rate:Q",
                        title="Retention Rate",
                        format=".1f",
                    ),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(
            retention_chart,
            use_container_width=True,
        )

    with right:
        st.subheader("Average Cohort Churn")

        churn_chart = (
            alt.Chart(churn_chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Month:Q",
                    title="Months Since Subscription",
                    axis=alt.Axis(tickMinStep=1),
                ),
                y=alt.Y(
                    "Churn Rate:Q",
                    title="Churn Rate (%)",
                    scale=alt.Scale(zero=True),
                ),
                tooltip=[
                    alt.Tooltip("Month:Q", title="Month", format=".0f"),
                    alt.Tooltip(
                        "Churn Rate:Q",
                        title="Churn Rate",
                        format=".1f",
                    ),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(
            churn_chart,
            use_container_width=True,
        )

    if (
        pd.notna(month_6_churn)
        and pd.notna(month_12_churn)
        and pd.notna(month_12_retention)
    ):
        st.info(
            f"Retention declines progressively during the first year. "
            f"Average cumulative churn reaches approximately "
            f"{month_6_churn:.1f}% by month 6 and "
            f"{month_12_churn:.1f}% by month 12, leaving average "
            f"12-month retention at {month_12_retention:.1f}%. "
            f"Later-tenure movements should be interpreted cautiously "
            f"because fewer mature cohorts contribute observations at "
            f"the longest subscription durations."
        )

    st.subheader("Churn at Key Milestones")

    display_milestones = milestones.copy()

    cohort_dates = pd.to_datetime(
        display_milestones.index,
        errors="coerce",
    )

    display_milestones.index = [
        date.strftime("%b %Y") if pd.notna(date) else str(original)
        for date, original in zip(cohort_dates, milestones.index)
    ]

    display_milestones.index.name = "Cohort"

    display_milestones = display_milestones.rename(
        columns={
            "month_1_churn": "Month 1 Churn (%)",
            "month_3_churn": "Month 3 Churn (%)",
            "month_6_churn": "Month 6 Churn (%)",
            "month_12_churn": "Month 12 Churn (%)",
        }
    )

    st.dataframe(
        display_milestones.round(2),
        use_container_width=True,
    )


# ==================================================
# TAB 3 — CUSTOMER SEGMENTS
# ==================================================

with tab3:
    st.subheader("Customer Segment Profiles")

    segment_col = (
        "cluster"
        if "cluster" in segments.columns
        else (
            "segment"
            if "segment" in segments.columns
            else None
        )
    )

    if segment_col is None:
        st.warning("No cluster or segment column was found.")

    else:
        # Evidence-based labels from the validated segment profiles
        # and the cluster-level retention comparison.
        segment_labels = {
            0: "High-Value Loyal",
            1: "Lower-Engagement Higher-Churn",
        }

        segment_display = segments.copy()

        segment_display["Customer Segment"] = (
            segment_display[segment_col]
            .map(segment_labels)
            .fillna(segment_display[segment_col].astype(str))
        )

        # --------------------------------------------------
        # SEGMENT SIZE METRICS
        # --------------------------------------------------

        segment_counts = segment_display["Customer Segment"].value_counts()
        total_segment_customers = len(segment_display)

        loyal_name = "High-Value Loyal"
        higher_churn_name = "Lower-Engagement Higher-Churn"

        loyal_count = int(segment_counts.get(loyal_name, 0))
        higher_churn_count = int(segment_counts.get(higher_churn_name, 0))

        loyal_share = (
            loyal_count / total_segment_customers * 100
            if total_segment_customers
            else 0
        )

        higher_churn_share = (
            higher_churn_count / total_segment_customers * 100
            if total_segment_customers
            else 0
        )

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "High-Value Loyal",
            f"{loyal_count:,}",
        )

        s2.metric(
            "Loyal Customer Share",
            f"{loyal_share:.1f}%",
        )

        s3.metric(
            "Higher-Churn Segment",
            f"{higher_churn_count:,}",
        )

        s4.metric(
            "Higher-Churn Share",
            f"{higher_churn_share:.1f}%",
        )

        # --------------------------------------------------
        # SEGMENT DISTRIBUTION
        # --------------------------------------------------

        distribution_data = (
            segment_display["Customer Segment"]
            .value_counts()
            .rename_axis("Customer Segment")
            .reset_index(name="Customers")
        )

        distribution_chart = (
            alt.Chart(distribution_data)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "Customer Segment:N",
                    title=None,
                    sort="-x",
                ),
                x=alt.X(
                    "Customers:Q",
                    title="Number of Customers",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Customer Segment:N",
                        title="Segment",
                    ),
                    alt.Tooltip(
                        "Customers:Q",
                        title="Customers",
                        format=",",
                    ),
                ],
            )
            .properties(height=220)
        )

        st.altair_chart(
            distribution_chart,
            use_container_width=True,
        )

        # --------------------------------------------------
        # SEGMENT PROFILE COMPARISON
        # --------------------------------------------------

        numeric_columns = (
            segments
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        profile_columns = [
            column
            for column in numeric_columns
            if column != segment_col
        ]

        segment_profile = (
            segments
            .groupby(segment_col)[profile_columns]
            .mean()
            .round(2)
        )

        segment_profile.index = [
            segment_labels.get(index, str(index))
            for index in segment_profile.index
        ]

        segment_profile.index.name = "Customer Segment"

        wanted_profile_columns = [
            "avg_session_duration",
            "avg_feature_usage",
            "avg_engagement_rate",
            "avg_satisfaction",
            "avg_nps",
            "avg_monthly_revenue",
            "latest_monthly_revenue",
            "active_months",
            "tenure_months",
        ]

        profile_display_columns = [
            column
            for column in wanted_profile_columns
            if column in segment_profile.columns
        ]

        profile_display = (
            segment_profile[profile_display_columns]
            .rename(
                columns={
                    "avg_session_duration": "Avg Session Duration",
                    "avg_feature_usage": "Avg Feature Usage",
                    "avg_engagement_rate": "Avg Engagement Rate",
                    "avg_satisfaction": "Avg Satisfaction",
                    "avg_nps": "Avg NPS",
                    "avg_monthly_revenue": "Avg Monthly Revenue",
                    "latest_monthly_revenue": "Latest Monthly Revenue",
                    "active_months": "Avg Active Months",
                    "tenure_months": "Avg Tenure Months",
                }
            )
        )

        st.subheader("Segment Comparison")

        st.dataframe(
            profile_display,
            use_container_width=True,
        )

        # --------------------------------------------------
        # DYNAMIC RETENTION BY SEGMENT
        # --------------------------------------------------

        required_retention_fields = {
            "customer_id",
            "activity_month",
            "cohort",
            "months_since_sub",
        }

        can_calculate_segment_retention = (
            required_retention_fields.issubset(master.columns)
            and "customer_id" in segments.columns
        )

        if can_calculate_segment_retention:
            master_segmented = master.merge(
                segments[["customer_id", segment_col]],
                on="customer_id",
                how="left",
            )

            master_segmented["activity_month"] = pd.to_datetime(
                master_segmented["activity_month"],
                errors="coerce",
            )

            master_segmented["cohort"] = pd.to_datetime(
                master_segmented["cohort"],
                errors="coerce",
            )

            valid_activity_dates = master_segmented["activity_month"].dropna()

            if not valid_activity_dates.empty:
                dataset_end = valid_activity_dates.max().to_period("M")

                customer_base = (
                    master_segmented[
                        ["customer_id", segment_col, "cohort"]
                    ]
                    .drop_duplicates("customer_id")
                    .dropna(subset=["cohort", segment_col])
                    .copy()
                )

                customer_base["cohort_month"] = (
                    customer_base["cohort"].dt.to_period("M")
                )

                retention_results = []

                for month in [3, 6, 12]:
                    eligible = customer_base[
                        customer_base["cohort_month"] <= dataset_end - month
                    ].copy()

                    active_at_month = (
                        master_segmented[
                            master_segmented["months_since_sub"] == month
                        ][["customer_id"]]
                        .drop_duplicates()
                        .assign(retained=1)
                    )

                    milestone_data = eligible.merge(
                        active_at_month,
                        on="customer_id",
                        how="left",
                    )

                    milestone_data["retained"] = (
                        milestone_data["retained"]
                        .fillna(0)
                        .astype(int)
                    )

                    milestone_summary = (
                        milestone_data
                        .groupby(segment_col)["retained"]
                        .agg(["mean", "count"])
                        .reset_index()
                    )

                    milestone_summary["Month"] = month
                    milestone_summary["Retention Rate"] = (
                        milestone_summary["mean"] * 100
                    )
                    milestone_summary["Churn Rate"] = (
                        100 - milestone_summary["Retention Rate"]
                    )

                    retention_results.append(
                        milestone_summary[
                            [
                                segment_col,
                                "Month",
                                "count",
                                "Retention Rate",
                                "Churn Rate",
                            ]
                        ]
                    )

                if retention_results:
                    retention_by_segment = pd.concat(
                        retention_results,
                        ignore_index=True,
                    )

                    retention_by_segment["Customer Segment"] = (
                        retention_by_segment[segment_col]
                        .map(segment_labels)
                        .fillna(
                            retention_by_segment[segment_col].astype(str)
                        )
                    )

                    st.subheader("Retention by Customer Segment")

                    retention_segment_chart = (
                        alt.Chart(retention_by_segment)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(
                                "Month:Q",
                                title="Months Since Subscription",
                                axis=alt.Axis(values=[3, 6, 12]),
                            ),
                            y=alt.Y(
                                "Retention Rate:Q",
                                title="Retention Rate (%)",
                                scale=alt.Scale(zero=False),
                            ),
                            color=alt.Color(
                                "Customer Segment:N",
                                title="Customer Segment",
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "Customer Segment:N",
                                    title="Segment",
                                ),
                                alt.Tooltip(
                                    "Month:Q",
                                    title="Month",
                                    format=".0f",
                                ),
                                alt.Tooltip(
                                    "Retention Rate:Q",
                                    title="Retention",
                                    format=".1f",
                                ),
                                alt.Tooltip(
                                    "Churn Rate:Q",
                                    title="Churn",
                                    format=".1f",
                                ),
                                alt.Tooltip(
                                    "count:Q",
                                    title="Eligible Customers",
                                    format=",",
                                ),
                            ],
                        )
                        .properties(height=350)
                    )

                    st.altair_chart(
                        retention_segment_chart,
                        use_container_width=True,
                    )

                    month_12_segment = (
                        retention_by_segment[
                            retention_by_segment["Month"] == 12
                        ]
                        .set_index(segment_col)
                    )

                    loyal_12_retention = (
                        month_12_segment.loc[0, "Retention Rate"]
                        if 0 in month_12_segment.index
                        else np.nan
                    )

                    higher_churn_12_retention = (
                        month_12_segment.loc[1, "Retention Rate"]
                        if 1 in month_12_segment.index
                        else np.nan
                    )

                    if (
                        pd.notna(loyal_12_retention)
                        and pd.notna(higher_churn_12_retention)
                    ):
                        loyal_12_churn = 100 - loyal_12_retention
                        higher_churn_12_churn = (
                            100 - higher_churn_12_retention
                        )

                        st.info(
                            f"The High-Value Loyal segment represents "
                            f"{loyal_share:.1f}% of customers and maintains "
                            f"approximately {loyal_12_retention:.1f}% "
                            f"retention at month 12 "
                            f"({loyal_12_churn:.1f}% churn). "
                            f"The larger Lower-Engagement Higher-Churn "
                            f"segment represents {higher_churn_share:.1f}% "
                            f"of customers and retains approximately "
                            f"{higher_churn_12_retention:.1f}% at month 12 "
                            f"({higher_churn_12_churn:.1f}% churn). "
                            f"The evidence indicates that retention strategy "
                            f"should particularly focus on improving "
                            f"engagement, feature adoption and value "
                            f"realization within the larger higher-churn segment."
                        )

        else:
            st.info(
                "Segment-level retention could not be calculated because "
                "one or more required fields are unavailable."
            )

        # --------------------------------------------------
        # CUSTOMER-LEVEL DATA
        # --------------------------------------------------

        st.subheader("Customer-Level Segment Data")

        customer_display = (
            segment_display
            .drop(columns=[segment_col])
        )

        ordered_columns = (
            ["customer_id", "Customer Segment"]
            + [
                column
                for column in customer_display.columns
                if column not in ["customer_id", "Customer Segment"]
            ]
        )

        customer_display = customer_display[ordered_columns]

        st.dataframe(
            customer_display.head(100),
            use_container_width=True,
            hide_index=True,
        )


# ==================================================
# TAB 4 — REVENUE & ENGAGEMENT
# ==================================================

with tab4:
    left, right = st.columns(2)

    # --------------------------------------------------
    # REVENUE
    # --------------------------------------------------

    with left:
        st.subheader("Average Monthly Revenue")

        if (
            "monthly_revenue" in master.columns
            and "activity_month" in master.columns
        ):
            revenue_data = (
                master
                .dropna(subset=["activity_month"])
                .groupby(
                    "activity_month",
                    as_index=False,
                )["monthly_revenue"]
                .mean()
                .sort_values("activity_month")
            )

            revenue_chart = (
                alt.Chart(revenue_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "activity_month:T",
                        title="Month",
                        axis=alt.Axis(
                            format="%b %Y",
                            labelAngle=-45,
                        ),
                    ),
                    y=alt.Y(
                        "monthly_revenue:Q",
                        title="Average Monthly Revenue",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "activity_month:T",
                            title="Month",
                            format="%b %Y",
                        ),
                        alt.Tooltip(
                            "monthly_revenue:Q",
                            title="Average Revenue",
                            format=",.2f",
                        ),
                    ],
                )
                .properties(height=350)
            )

            st.altair_chart(
                revenue_chart,
                use_container_width=True,
            )

        else:
            st.info("Revenue fields are not available.")

    # --------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------

    with right:
        st.subheader("Average Engagement")

        engagement_field = None
        engagement_title = None

        if "response_rate" in master.columns:
            engagement_field = "response_rate"
            engagement_title = "Average Response Rate"

        elif "engagement_count" in master.columns:
            engagement_field = "engagement_count"
            engagement_title = "Average Engagement Count"

        if (
            engagement_field is not None
            and "activity_month" in master.columns
        ):
            engagement_data = (
                master
                .dropna(subset=["activity_month"])
                .groupby(
                    "activity_month",
                    as_index=False,
                )[engagement_field]
                .mean()
                .sort_values("activity_month")
            )

            engagement_chart = (
                alt.Chart(engagement_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "activity_month:T",
                        title="Month",
                        axis=alt.Axis(
                            format="%b %Y",
                            labelAngle=-45,
                        ),
                    ),
                    y=alt.Y(
                        f"{engagement_field}:Q",
                        title=engagement_title,
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "activity_month:T",
                            title="Month",
                            format="%b %Y",
                        ),
                        alt.Tooltip(
                            f"{engagement_field}:Q",
                            title=engagement_title,
                            format=".2f",
                        ),
                    ],
                )
                .properties(height=350)
            )

            st.altair_chart(
                engagement_chart,
                use_container_width=True,
            )

        else:
            st.info("Engagement fields are not available.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption("GrowthFlow Customer Retention Strategy Optimization")
