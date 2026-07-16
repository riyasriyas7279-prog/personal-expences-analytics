import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. UI SETUP & PAGE ARCHITECTURE
# ==========================================
st.set_page_config(
    page_title="Personal Expense Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Personal Expense Analytics Dashboard")
st.markdown("### An automated intelligence pipeline tracking outlays & budgets")
st.write("---")

# ==========================================
# 2. SEED SYNTHETIC DATA FOR PORTFOLIO VALIDATION
# ==========================================
DATA_FILE = "personal_expenses.csv"


def generate_mock_data():
    """Generates a structured multi-month dataframe for testing."""
    np.random.seed(101)
    dates = pd.date_range(start="2026-01-01", end="2026-07-15", freq="D")
    categories = [
        "Food & Dining",
        "Utilities",
        "Rent",
        "Entertainment",
        "Travel",
        "Healthcare",
    ]

    mock_records = []
    for current_date in dates:
        for _ in range(np.random.randint(1, 4)):
            category = np.random.choice(
                categories, p=[0.35, 0.15, 0.05, 0.20, 0.15, 0.10]
            )

            if category == "Rent":
                amount = 1200.0 if current_date.day == 1 else 0.0
            elif category == "Utilities":
                amount = (
                    np.random.uniform(40, 150)
                    if current_date.day in [5, 12]
                    else 0.0
                )
            else:
                amount = np.random.exponential(scale=35.0)

            if amount > 0:
                mock_records.append(
                    {
                        "Date": current_date.strftime("%Y-%m-%d"),
                        "Category": category,
                        "Amount": round(amount, 2),
                        "Description": f"Simulated {category} charge",
                    }
                )

    pd.DataFrame(mock_records).to_csv(DATA_FILE, index=False)


if not os.path.exists(DATA_FILE):
    generate_mock_data()


# ==========================================
# 3. PANDAS ETL PIPELINE
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    # Standard format for readable drop-down displaying
    df["Month"] = df["Date"].dt.strftime("%Y - %B")
    # Store dynamic baseline components for chronological sorting
    df["Month_Sort"] = df["Date"].dt.to_period("M")
    df["Category"] = df["Category"].str.strip().str.title()
    return df


df_clean = load_and_preprocess_data()

# ==========================================
# 4. SIDEBAR CONFIGURATION & CHRONOLOGICAL FILTERS
# ==========================================
st.sidebar.header("⚙️ Configuration & Budgets")

# Chronological sorting map logic
unique_months = (
    df_clean[["Month", "Month_Sort"]]
    .drop_duplicates()
    .sort_values(by="Month_Sort")
)
available_months = unique_months["Month"].tolist()
selected_month = st.sidebar.selectbox(
    "Select Target Analytics Month", available_months
)

# Expandable allocation layout
with st.sidebar.expander("🛠️ Set Category Budget Thresholds", expanded=True):
    budget_limits = {
        "Food & Dining": st.number_input("Food & Dining Budget ($)", value=450),
        "Entertainment": st.number_input("Entertainment Budget ($)", value=300),
        "Travel": st.number_input("Travel Budget ($)", value=400),
        "Utilities": st.number_input("Utilities Budget ($)", value=200),
        "Healthcare": st.number_input("Healthcare Budget ($)", value=250),
        "Rent": st.number_input("Rent Budget ($)", value=1300),
    }

# Filter dataset context
df_month = df_clean[df_clean["Month"] == selected_month]

# ==========================================
# 5. CORE KPI CALCULATIONS & ANOMALY DETECTION
# ==========================================
total_spend = df_month["Amount"].sum()
avg_transaction = df_month["Amount"].mean()
highest_spend_row = (
    df_month.loc[df_month["Amount"].idxmax()]
    if not df_month.empty
    else {"Category": "N/A", "Amount": 0}
)

# Transaction Anomaly Logic: Flags entries 3 standard deviations above the historical category mean
anomalous_transactions = []
if not df_month.empty:
    category_means = df_clean.groupby("Category")["Amount"].mean()
    category_stds = df_clean.groupby("Category")["Amount"].std().fillna(0)

    for idx, row in df_month.iterrows():
        cat = row["Category"]
        amt = row["Amount"]
        # Trigger flag threshold rule
        threshold = category_means.get(cat, 0) + (3 * category_stds.get(cat, 0))
        if amt > threshold and amt > 100:  # Avoid flags on minor small anomalies
            anomalous_transactions.append(row)

# Render Core KPI Cards
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("💰 Total Monthly Outlay", f"${total_spend:,.2f}")
kpi2.metric("📉 Mean Transaction Ticket", f"${avg_transaction:,.2f}")
kpi3.metric(
    "🔥 Max Single Outlier",
    f"${highest_spend_row['Amount']:,.2f}",
    delta=f"Category: {highest_spend_row['Category']}",
    delta_color="off",
)

st.write("---")

# ==========================================
# 6. BUSINESS INTELLIGENCE & VISUAL ANALYTICS
# ==========================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### 📈 Categorical Spend vs. Allocations")
    category_summary = (
        df_month.groupby("Category")["Amount"].sum().reset_index()
    )

    # Process budget breach warnings
    alerts = []
    category_summary["Budget"] = category_summary["Category"].map(budget_limits)

    for idx, row in category_summary.iterrows():
        cat = row["Category"]
        amt = row["Amount"]
        limit = row["Budget"]
        if pd.notna(limit) and amt > limit:
            alerts.append(
                f"⚠️ **{cat}** crossed allocation limit! Spent **${amt:,.2f}** / Max: ${limit:,.2f}"
            )

    # Advanced Charting: Grouped comparison using Plotly Graph Objects
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            name="Actual Spend",
            x=category_summary["Category"],
            y=category_summary["Amount"],
            marker_color="#4a90e2",
            text=category_summary["Amount"].round(0),
            textposition="auto",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            name="Budget Limit",
            x=category_summary["Category"],
            y=category_summary["Budget"],
            marker_color="#ff6b6b",
            opacity=0.6,
        )
    )
    fig_bar.update_layout(
        barmode="group", height=380, margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown("#### 🍕 Proportional Outlay Breakdown")
    fig_pie = px.pie(
        category_summary,
        values="Amount",
        names="Category",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# 7. ANOMALY DETECTION ENGINE & SYSTEM LOGS
# ==========================================
st.markdown("#### 🛡️ Budget Compliance & Anomaly Diagnostics")
diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.markdown("##### Budget Threshold Reports")
    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("🎉 All category streams operating within normal limits.")

with diag_col2:
    st.markdown("##### Statistical Spike Transcripts")
    if anomalous_transactions:
        for transaction in anomalous_transactions:
            st.warning(
                f"🚨 **Spike Detected:** Spent **${transaction['Amount']:,.2f}** on *{transaction['Description']}* ({transaction['Date'].strftime('%Y-%m-%d')})"
            )
    else:
        st.info("ℹ️ No statistical single transaction anomalies found this month.")

st.write("---")

# ==========================================
# 8. GRANULAR TRANSACTION LEDGER
# ==========================================
st.markdown("#### 🔍 Granular Transaction Ledger")
st.dataframe(
    df_month[["Date", "Category", "Amount", "Description"]].sort_values(
        by="Date", ascending=False
    ),
    use_container_width=True,
)