import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# PAGE SETUP

st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    layout="wide"
)

# LOAD FILES

@st.cache_data
def load_data():
    return pd.read_csv("customer_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("catboost.pkl")

@st.cache_resource
def load_features():
    return joblib.load("feature_cols.pkl")

df = load_data()
model = load_model()
feature_cols = load_features()

df["risk_score"] = model.predict_proba(
    df[feature_cols]
)[:,1]

risk_bucket = pd.cut(
    df["risk_score"],
    bins=[0,0.3,0.7,1],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)
df["behavioral_churn"] = (
    (
        (df["activity_ratio"] <= 0.10).astype(int)
        +
        (df["recency_at_end_of_2017"] >= 6).astype(int)
        +
        (df["booking_trend"] <= 0).astype(int)
        +
        (df["points_per_flight"] < 500).astype(int)
    ) >= 3
).astype(int)

# SIDEBAR

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "",
    [
        "Executive Dashboard",
        "Churn Insights",
        "Customer Segmentation",
        "Customer Prediction",
        "Retention Strategy"
    ]
)

# EXECUTIVE DASHBOARD

if page == "Executive Dashboard":

    st.title("✈️ Airline Loyalty Intelligence Dashboard")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Customers",
        f"{len(df):,}"
    )

    c2.metric(
        "Actual Churn",
        f"{df['churn'].mean()*100:.2f}%"
    )

    c3.metric(
        "Behavioral Churn",
        f"{df['behavioral_churn'].mean()*100:.2f}%"
    )

    c4.metric(
        "Average CLV",
        f"${df['CLV'].mean():,.0f}"
    )

    st.divider()

    col1,col2 = st.columns(2)

    with col1:

        st.subheader(
            "Customer Risk Distribution"
        )

        st.bar_chart(
            risk_bucket.value_counts()
        )

    with col2:

        st.subheader(
            "Top Drivers Of Churn"
        )

        imp = model.get_feature_importance()

        feat_imp = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": imp
        })

        feat_imp = feat_imp.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        st.bar_chart(
            feat_imp.set_index(
                "Feature"
            )
        )

    st.divider()

    st.subheader(
        "Top High Value Customers At Risk"
    )

    top_risk = (
        df.sort_values(
            ["risk_score","CLV"],
            ascending=False
        )
        [
            [
                "Loyalty Number",
                "CLV",
                "risk_score",
                "Province"
            ]
        ]
        .head(20)
    )

    st.dataframe(
        top_risk,
        use_container_width=True
    )

# ===============================
# CHURN INSIGHTS

elif page == "Churn Insights":

    st.title(
        "📊 Business Insights"
    )
##Insight 1
    st.subheader(
        "Insight 1: Reward Redemption Reduces Churn"
    )

    redeem = pd.DataFrame({
        "Customer Type":[
            "Retained",
            "Churned"
        ],
        "Redeem Ratio":[
            0.267,
            0.013
        ]
    })

    st.bar_chart(
        redeem.set_index(
            "Customer Type"
        )
    )

    st.success(
        "Customers actively redeeming rewards show significantly lower churn."
    )

    st.divider()
## Insight 2
    st.subheader(
        "Insight 2: Enrollment Cohort Analysis"
    )

    cohort = (
        df.groupby(
            "Enrollment Year"
        )["churn"]
        .mean()
    )

    st.line_chart(
        cohort
    )

    st.info(
        "2017 customer cohort experienced the highest churn."
    )

    st.divider()
## Insight 3
    st.subheader(
        "Insight 3: Province Wise Churn"
    )

    province = (
        df.groupby(
            "Province"
        )["churn"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        province
    )

    st.info(
        "Prince Edward Island and Yukon exhibit the highest churn rates."
    )

    st.divider()
 ## Insight 4
    st.subheader(
        "Insight 4: Behavioral Churn Early Warning"
    )

    st.code(
"""
Activity Ratio <= 0.10
+
Recency >= 6 months
+
Booking Trend <= 0
+
Points Per Flight < 500

>= 3 conditions
= Behavioral Churn
"""
    )

    cross = pd.crosstab(
        df["behavioral_churn"],
        df["churn"]
    )

    st.dataframe(
        cross
    )

    st.success(
        "Behavioral churn successfully identifies disengaged customers before cancellation."
    )

    st.divider()

    st.subheader(
        "Insight 5: Loyalty Tier Impact"
    )

    loyalty = (
        df.groupby(
            "Loyalty Card"
        )["activity_ratio"]
        .mean()
    )

    st.bar_chart(
        loyalty
    )

    st.info(
        "Loyalty tier alone is not a strong predictor of churn."
    )
    # Insight 2
    
    st.subheader(
        "Insight 6: Recency Is The Strongest Churn Signal"
    )

    st.info("""
    Customers who have remained inactive for
    longer periods exhibit substantially higher
    churn risk.
    """)

    if "recency_at_end_of_2017" in df.columns:

        recency_analysis = (
            df.groupby("churn")
            ["recency_at_end_of_2017"]
            .mean()
        )

        st.bar_chart(recency_analysis)

    st.markdown("---")
    
    # Insight 7
    st.subheader(
        "Insight 7: Activity Ratio Strongly Impacts Retention"
    )

    st.info("""
    Customers with higher engagement levels
    and frequent activity are significantly
    less likely to churn.
    """)

    activity_analysis = (
        df.groupby("churn")
        ["activity_ratio"]
        .mean()
    )

    st.bar_chart(activity_analysis)

    st.markdown("---")

    # Insight 8
   
    st.subheader(
        "Insight 8: Booking Trend Reflects Future Risk"
    )

    st.info("""
    Declining booking trends often indicate
    decreasing customer engagement and
    future churn risk.
    """)

    booking_analysis = (
    df.groupby("churn")
    ["booking_trend"]
    .mean()
    )


    st.bar_chart(booking_analysis)

    st.markdown("---")
 ##Insight 9
   
    st.subheader(
        "Insight 9: Seasonality Is Relatively Weak"
    )

    st.info("""
    February-March observed the lowest booking
    activity while June-July showed peak activity.

    However, behavioural features such as recency,
    activity ratio and booking trend were much
    stronger predictors of churn.
    """)

    season_df = pd.DataFrame({
        "Month": ["Feb", "Mar", "Jun", "Jul"],
        "Relative Activity": [1, 1.2, 5, 5.5]
    })

    st.line_chart(
        season_df.set_index("Month")
    )


# CUSTOMER SEGMENTATION

elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.markdown("""
    ### Segment Definitions

    #### 🟢 Active Loyalists

    - High activity ratio
    - Frequent bookings
    - Low recency
    - Strong loyalty engagement

    #### 🟡 At-Risk Customers

    - Declining activity
    - Moderate inactivity
    - Reduced booking frequency

    #### 🔴 Dormant Customers

    - Long inactivity periods
    - Very low engagement
    - High churn risk
    """)

    st.markdown("---")

    st.subheader("Activity Ratio Distribution")

    import plotly.express as px

    fig = px.histogram(
        df,
        x="activity_ratio",
        nbins=20,
        title="Activity Ratio Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.info("""Business Insight:
            Active customers are substantially less likely to leave the loyalty program.
                Low activity levels often appear months before actual churn occurs.""")
    st.subheader("Booking Trend Distribution")

    fig = px.histogram(
        df,
        x="booking_trend",
        nbins=30,
        title="Booking Trend Distribution"
)

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.info("""Business Insight:

Customers inactive for long periods show the highest churn propensity.
Recency acts as an early behavioural signal and should be monitored continuously.""")

# =====================================================
# CUSTOMER PREDICTION
# =====================================================
elif page == "Customer Prediction":

    st.title("🔮 Customer Churn Prediction")

    customer_id = st.selectbox(
        "Select Customer",
        sorted(df["Loyalty Number"].unique())
    )

    customer = df[
        df["Loyalty Number"] == customer_id
    ]

    st.subheader("Customer Profile")

    st.dataframe(customer)

    if st.button("Predict Churn"):

        X_pred = customer[feature_cols]

        prob = model.predict_proba(X_pred)[0][1]

        st.metric(
            "Churn Probability",
            f"{prob*100:.2f}%"
        )

        st.progress(float(prob))

        if prob >= 0.70:

            st.error(
                "🔴 HIGH RISK CUSTOMER"
            )

        elif prob >= 0.40:

            st.warning(
                "🟡 MEDIUM RISK CUSTOMER"
            )

        else:

            st.success(
                "🟢 LOW RISK CUSTOMER"
            )
    st.subheader(
        "Is This Customer At Risk Due To Behavioral Signals?"
    )

    reasons = []

    if customer["activity_ratio"].iloc[0] < 0.10:
        reasons.append(
            "Low Activity Ratio"
        )

    if customer["recency_at_end_of_2017"].iloc[0] > 6:
        reasons.append(
            "Long Period Of Inactivity"
        )

    if customer["booking_trend"].iloc[0] < 0:
        reasons.append(
            "Declining Booking Trend"
        )

    if customer["points_per_flight"].iloc[0] < 500:
        reasons.append(
            "Low Loyalty Engagement"
        )

    if len(reasons)==0:
        st.success(
            "No major churn signals detected."
        )

    for r in reasons:
        st.warning(r)

# RETENTION STRATEGY

elif page == "Retention Strategy":

    st.title("🎯 Retention Strategy Engine")

    customer_id = st.selectbox(
        "Select Customer",
        sorted(df["Loyalty Number"].unique()),
        key="retention"
    )

    customer = df[
        df["Loyalty Number"] == customer_id
    ]

    X_pred = customer[feature_cols]

    prob = model.predict_proba(X_pred)[0][1]
    clv = customer["CLV"].iloc[0]

    actions = []

    if customer["recency_at_end_of_2017"].iloc[0] > 6:
        actions.append(
            "Launch Win-Back Campaign"
        )

    if customer["redeem_ratio"].iloc[0] < 0.05:
        actions.append(
            "Offer Bonus Miles"
        )

    if customer["activity_ratio"].iloc[0] < 0.10:
        actions.append(
            "Send Personalized Travel Offer"
        )

    if clv > 10000:
        actions.append(
            "Assign Premium Retention Manager"
        )

    st.metric(
        "Churn Probability",
        f"{prob*100:.2f}%"
    )

    st.subheader(
        "Recommended Business Action"
    )

    if prob >= 0.70:

        st.error(
            "High Risk Customer"
        )

        st.write("""
        ### Recommended Actions

        - 5000 Bonus Miles
        - Personalized Retention Offer
        - Dedicated Retention Campaign
        - Email + SMS Follow-Up
        - Special Loyalty Rewards
        """)

    elif prob >= 0.40:

        st.warning(
            "Medium Risk Customer"
        )

        st.write("""
        ### Recommended Actions

        - Discount Voucher
        - Seasonal Promotions
        - Loyalty Engagement Campaign
        - Bonus Point Incentives
        """)

    else:

        st.success(
            "Low Risk Customer"
        )

        st.write("""
        ### Recommended Actions

        - Tier Upgrade Opportunities
        - Premium Benefits
        - Exclusive Offers
        - Continued Engagement Programs
        """)
