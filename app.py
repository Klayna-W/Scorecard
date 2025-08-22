import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from datetime import datetime


#------------------------------------------------- DONEEEEEE --------------------------------------------
#------------------------------------------------- DONEEEEEE --------------------------------------------
#------------------------------------------------- DONEEEEEE --------------------------------------------

st.set_page_config(page_title="Manager Dashboard", layout="wide",initial_sidebar_state="collapsed")

# -----------------------------
# --- Session State ---
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True  # Assume already logged in for simplicity
if "username" not in st.session_state:
    st.session_state.username = "admin"
if "comments" not in st.session_state:
    st.session_state.comments = {}




# -----------------------------
# --- Load Data ---
# -----------------------------
employees = pd.read_csv("employees.csv", parse_dates=["Hire Date"])
LAE_Metrics = pd.read_csv("NewData.csv", parse_dates=["Date"])
Hours = pd.read_csv("Hours.csv", parse_dates=["Date"])




# Fill missing managers
daily_metrics = daily_metrics.merge(
    employees[['Employee', 'Employee Name', 'Position', 'Manager', 'Hire Date', 'Office']],
    left_on='Agent',
    right_on='Employee',
    how='left'
)
daily_metrics['Manager'] = daily_metrics['Manager'].fillna("Unknown")
daily_metrics['Employee Name'] = daily_metrics['Employee Name'].fillna(daily_metrics['Agent'])



# -----------------------------
# --- Sidebar: Filters & Logout ---
# -----------------------------
employees = employees[employees['Office'] != 'Irvine']
office_managers_df = employees[employees['Position'] == "Office Manager"][['Employee Name', 'Office']]

# For dropdown display, create a combined label (e.g., "Jane Doe - New York")
office_managers_df['Label'] = office_managers_df['Employee Name'] + ' - ' + office_managers_df['Office']

with st.sidebar.expander("Filters 🔽", expanded=False):
    selected_label = st.selectbox("Select Office Manager", sorted(office_managers_df['Label']))
    st.button("Logout")

# Optional: To retrieve actual manager name and location from the selection
selected_row = office_managers_df[office_managers_df['Label'] == selected_label].iloc[0]
selected_manager = selected_row['Employee Name']
selected_office = selected_row['Office']


# Filter data for selected manager
manager_office = employees.loc[employees['Employee Name'] == selected_manager, 'Office'].iloc[0]
LAE_filtered = LAE_Metrics[LAE_Metrics['Office'] == manager_office]
Hours_filtered = LAE_Metrics[LAE_Metrics['Office'] == manager_office]

 

# ----------------------------- # --- Users / Credentials --- # ----------------------------- 
users = { "admin": {"name": "admin", "password": "a"}, 
         "manager1": {"name": "John Doe", "password": "password1"} }
 # ----------------------------- # --- Login --- # ----------------------------- 
if not st.session_state.logged_in: 
    st.title("🔒 Manager Dashboard Login") 
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password") 
    login = st.button("Login") # if login: 
    if username_input in users and password_input == users[username_input]["password"]:
        st.session_state.logged_in = True
        st.session_state.username = username_input # st.experimental_rerun() 
    else: # st.error("Incorrect username or password") 
        st.stop()



# -----------------------------
# --- Manager HireDate ---
# -----------------------------
# Manager hire date
manager_hire_date = employees.loc[
    (employees['Position'] == "Office Manager") & (employees['Employee Name'] == selected_manager),'Hire Date'].min()


if pd.notna(manager_hire_date):
    today = pd.Timestamp.today()
    years = today.year - manager_hire_date.year
    months = today.month - manager_hire_date.month
    if months < 0:
        years -= 1
        months += 12
    tenure_text = f"{years} years, {months} months (since {manager_hire_date.date()})"
else:
    tenure_text = "Hire date not available"


# -----------------------------
# --- Manager / Office Info ---
# -----------------------------

# Get office name from selected manager
def count_positions(df, office, positions):
    return df[
        (df['Office'] == office) &
        (df['Position'].isin(positions))
    ]['Employee'].nunique()

count_agents = ['Floor Assistant', 'Agent']
count_setters = ['Jr Setter', 'Sr Setter']
num_employees_sales = count_positions(employees, manager_office, count_agents)
num_employees_csr = count_positions(employees, manager_office, count_setters)
manager_row = employees[employees['Employee Name'] == selected_manager]
manager_office = manager_row['Office'].iloc[0]  # Office corresponding to manager

# -----------------------------
# --- Header ---
# -----------------------------
st.title(f"📊 {selected_office} Dashboard")
st.markdown(f"**Manager:** {selected_manager} | **Time Worked:** {tenure_text}")
st.markdown(f"**Current Agents:** {num_employees_sales} | **Setters:** {num_employees_csr} ")

st.markdown("---")


import streamlit as st
import pandas as pd
import altair as alt

# --- Monthly aggregation ---
LAE_filtered['MonthD'] = LAE_filtered['Date'].dt.to_period('M')

manager_monthly_summary = (
    LAE_filtered[LAE_filtered['Office'] == manager_office]
    .groupby('MonthD')
    .agg(
        NB=('NB', 'sum'),
        GI=('GI', 'sum'),
        BF=('BF', 'sum'),
        # Immigration=('Immigration', 'sum'),
        DMV_count=('DMV', 'sum'),
        # ImmigrationSum=('Immigration $', 'sum'),
        DMV=('DMV $', 'sum')
    )
    .sort_index(ascending=True)
    .T
)
# # --- Reset index and add Metric column ---
# manager_monthly_summary.index.name = "Metric"
# manager_monthly_summary = manager_monthly_summary.reset_index()

# --- Compute projections ---
today = pd.Timestamp.today() - pd.Timedelta(days=1)
current_month = today.to_period("M")

all_days = pd.date_range(start=current_month.start_time, end=current_month.end_time)
working_days = all_days[all_days.dayofweek != 6]  # exclude Sundays
total_working_days = len(working_days)
working_days_so_far = len(working_days[working_days <= today])

# Identify current month column
current_col = None
for col in manager_monthly_summary.columns:
    if str(current_month) in str(col):
        current_col = col
        break

if current_col:
    manager_monthly_summary["Projection"] = ((
        manager_monthly_summary[current_col] / 11) * total_working_days
    ).round(0)
else:
    manager_monthly_summary["Projection"] = 0

bf_percent = (manager_monthly_summary.loc['BF'] / manager_monthly_summary.loc['GI'] * 100).round(0)
manager_monthly_summary.loc['NB %'] = bf_percent

dmv_percent = (manager_monthly_summary.loc['DMV_count'] / manager_monthly_summary.loc['NB'] * 100).round(0)
manager_monthly_summary.loc['DMV %'] = dmv_percent




# --- Step 1: Create Top Payee per Month per Office ---
nb_summary = (
    LAE_filtered
    .groupby(['Office', 'MonthD', 'Payee'])
    .agg(NB_count=('NB', 'count'))  # or 'sum' if NB is numeric
    .reset_index()
)

# Get Payee with max NB for each Office + Month
idx = nb_summary.groupby(['Office', 'MonthD'])['NB_count'].idxmax()
top_payee_per_month = nb_summary.loc[idx].reset_index(drop=True)


# --- Step 3: Build dictionary of Month -> Top Payee ---
top_payee_row = {
    month: top_payee_per_month.loc[top_payee_per_month['MonthD'] == month, 'Payee'].values[0]
    if month in top_payee_per_month['MonthD'].values else ""
    for month in manager_monthly_summary.columns
}

# --- Step 4: Add Top Payee as a new row in your existing table ---
manager_monthly_summary.loc['Top Payee'] = pd.Series(top_payee_row)


# --- Reset index and add Metric column ---
manager_monthly_summary.index.name = "Metric"
manager_monthly_summary = manager_monthly_summary.reset_index()


Hours['Overtime'] = Hours['Overtime'].str.replace('-', '0')
Hours['MealPenalty'] = Hours['MealPenalty'].str.replace('-', '0')
Hours['MealPenalty'] = pd.to_numeric(Hours['MealPenalty'])
Hours['Overtime'] = pd.to_numeric(Hours['Overtime'])

# --- Step 1: Prepare hours_summary ---
Hours['MonthD'] = Hours['Date'].dt.to_period('M')
hours_summary = (
    Hours[Hours['Office'] == manager_office]
    .groupby('MonthD')
    .agg(OT=('Overtime', 'sum'), MealP=('MealPenalty', 'sum'))
)

# Transpose to match manager_monthly_summary format
hours_summary_t = hours_summary.T
hours_summary_t.index.name = 'Metric'
hours_summary_t = hours_summary_t.reset_index()


# --- Step 3: Merge both tables ---
manager_monthly_summary = pd.concat([manager_monthly_summary, hours_summary_t], axis=0, sort=False)



# --- Table formatting ---
metric_formats = {
    "NB": "{:,.0f}",
    "GI": "${:,.0f}",
    "BF": "${:,.0f}",
    "NB %": "{:,.0f}%","DMV %": "{:,.0f}%",
    # "Immigration": "{:,.0f}",
    "DMV_count": "{:,.0f}",  
    #   "Immigration $": "{:,.0f}",
    "DMV": "${:,.0f}",

}

def format_row(row):
    metric = row["Metric"]
    fmt = metric_formats.get(metric, "{:,.0f}")
    return [fmt.format(x) if pd.notna(x) and not isinstance(x, str) else x for x in row]

formatted_table = manager_monthly_summary.copy()
formatted_table.iloc[:, 1:] = formatted_table.apply(format_row, axis=1, result_type="expand").iloc[:, 1:]










st.subheader(" Totals by Month")
st.dataframe(formatted_table, use_container_width=True)

# --- Prepare charting data ---
# Convert month columns to strings
month_cols = [col for col in manager_monthly_summary.columns if col not in ["Metric", "Projection"]]
manager_monthly_summary.rename(columns={col: str(col) for col in month_cols}, inplace=True)

chart_df = manager_monthly_summary.set_index("Metric")

# Replace current month column with Projection
if current_col:
    chart_df[str(current_month)] = chart_df["Projection"]

# Drop Projection column (already included in current month)
chart_df = chart_df.drop(columns=["Projection"])

# Sort months chronologically
month_cols = [c for c in chart_df.columns if c != "Metric"]
month_periods = [pd.Period(c) for c in month_cols]
sorted_months = [str(p) for p in sorted(month_periods)]
chart_df = chart_df[sorted_months]
# --- Multi-metric selection ---
selected_metrics = st.multiselect(
    "Select metrics to plot",
    chart_df.index.tolist(),
    default=["NB", "GI"]
)

if selected_metrics:
    n = len(selected_metrics)
    # Create exactly n columns (all in one row)
    cols = st.columns(n)

    for i, metric in enumerate(selected_metrics):
        df = chart_df.loc[metric].reset_index()
        df.columns = ["Month", "Value"]

        # Format month for X-axis
        df["Month"] = df["Month"].apply(lambda x: pd.Period(x).strftime("%b %Y") if not isinstance(x, str) else x)

        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("Month:N", title="Month", sort=df["Month"].tolist()),
            y=alt.Y("Value:Q", title=metric),
            tooltip=["Month", "Value"]
        ).properties(title=metric, width=300, height=400)  # fixed width/height

        cols[i].altair_chart(chart, use_container_width=True)












