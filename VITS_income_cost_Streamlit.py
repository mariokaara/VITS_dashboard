import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Streamlit title
st.title('VITS Dashboard')

# Configuration for the first graph
window = 3  # Rolling months
smoothness = 2  # Smoothness (k)

# Assuming 'vits_data.xlsx' is updated and located in a known directory
file_path = 'vits_data.xlsx'  # Update this with the path to your Excel file

# Selecting country for visitors data
country = st.selectbox('Select Country:', ('EE', 'PL'), index=1)  # Default to 'PL'

# Load and prepare the revenue and expenses data
@st.cache_data
def load_rev_exp_data(file_path):
    df_rev_exp = pd.read_excel(file_path, sheet_name='rev_exp', usecols='B:AK', skiprows=2, nrows=2, header=None)
    df_rev_exp = df_rev_exp.transpose()
    df_rev_exp.columns = ['Expenses', 'Revenue']
    df_rev_exp = df_rev_exp.apply(pd.to_numeric, errors='coerce')
    df_rev_exp.index = pd.date_range(start='01-2022', periods=len(df_rev_exp), freq='M')
    return df_rev_exp

df_rev_exp = load_rev_exp_data(file_path)
df_rolling = df_rev_exp.rolling(window=window).mean()
df_rolling_filled = df_rolling.ffill().bfill()

# Plotting revenue and expenses
st.subheader('Expenses and Revenue Overview')
fig, ax = plt.subplots(figsize=(15, 9))
dates_num = mdates.date2num(df_rolling_filled.index)
projection_start_date = mdates.date2num(pd.Timestamp('2024-04-01'))

# Smoothing
cost_spline = make_interp_spline(dates_num, df_rolling_filled['Expenses'], k=smoothness)
income_spline = make_interp_spline(dates_num, df_rolling_filled['Revenue'], k=smoothness)
smooth_dates = np.linspace(dates_num.min(), dates_num.max(), 300)
smooth_cost = cost_spline(smooth_dates)
smooth_income = income_spline(smooth_dates)

# Historical and projection plotting
ax.plot_date(smooth_dates[smooth_dates < projection_start_date], smooth_cost[smooth_dates < projection_start_date], '-', label='Expenses (Historical)', color='red')
ax.plot_date(smooth_dates[smooth_dates < projection_start_date], smooth_income[smooth_dates < projection_start_date], '-', label='Revenue (Historical)', color='green')
ax.plot_date(smooth_dates[smooth_dates >= projection_start_date], smooth_cost[smooth_dates >= projection_start_date], '--', label='Expenses (Projection)', color='red')
ax.plot_date(smooth_dates[smooth_dates >= projection_start_date], smooth_income[smooth_dates >= projection_start_date], '--', label='Revenue (Projection)', color='green')

# Filling
ax.fill_between(smooth_dates, smooth_cost, smooth_income, where=(smooth_cost >= smooth_income), color='lightcoral', interpolate=True, label='Expenses > Revenue')
ax.fill_between(smooth_dates, smooth_cost, smooth_income, where=(smooth_cost <= smooth_income), color='lightgreen', interpolate=True, label='Revenue > Expenses')

ax.legend()
ax.grid(True)
ax.set(title=f'Expenses and Revenue Rolling {window}-Month Average', xlabel='Date', ylabel='Value')
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
st.pyplot(fig)

# Load and prepare the visitor data
@st.cache_data
def load_visitor_data(file_path, country):
    df_visitors = pd.read_excel(file_path, sheet_name=f'visitors_{country}')
    df_visitors['Month'] = pd.to_datetime(df_visitors['Month'], format='%b-%y')
    df_visitors.sort_values('Month', inplace=True)
    return df_visitors

df_visitors = load_visitor_data(file_path, country)

# Plotting visitor data
st.subheader(f'Monthly New Visitors and Sign-ups for {country}')
fig, ax1 = plt.subplots(figsize=(14, 7))
color = 'tab:blue'
ax1.set_xlabel('Month')
ax1.set_ylabel('New Visitors', color=color)
ax1.plot(df_visitors['Month'].dt.strftime('%b-%y'), df_visitors['New visitors'], label='New Visitors', color=color, marker='x')
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='x', rotation=45)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Sign-ups', color=color) 
ax2.plot(df_visitors['Month'].dt.strftime('%b-%y'), df_visitors['Sign-ups'], label='Sign-ups', color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
st.pyplot(fig)

# Plot Signups/1K new visitors
st.subheader(f'Signups per 1K New Visitors in {country}')
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(df_visitors['Month'], df_visitors['Signups/1K new visitors'], color='skyblue')
ax.set_title('Signups per 1K New Visitors')
ax.set_xlabel('Month')
ax.set_ylabel('Signups per 1K New Visitors')
plt.xticks(rotation=45)
st.pyplot(fig)
