import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import io
from datetime import datetime, timedelta
from prediction import run_full_pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Borivali Groundwater Analytics", layout="wide")
uploaded_file = r"BORIVALI_DWLR_REALTIME_multi.csv"

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if 'extraction_mcm' not in df.columns and 'demand_mcm' in df.columns:
        df['extraction_mcm'] = df['demand_mcm'] * 0.85 
    elif 'extraction_mcm' not in df.columns:
        df['extraction_mcm'] = 0 

    def get_season(month):
        if month in [6, 7, 8, 9]: return 'Monsoon'
        if month in [10, 11]: return 'Post-Monsoon'
        if month in [12, 1, 2]: return 'Winter'
        return 'Summer'

    df['season'] = df['month'].apply(get_season)

    def get_status(level):
        if level < 5: return 'Safe'
        if level < 10: return 'Semi-Critical'
        return 'Critical'

    df['status'] = df['water_level_m'].apply(get_status)
    return df

def load_predictions():
    # Note: We don't cache this strictly if we want to see new results after every pipeline run
    pred_df = pd.read_csv('24hrs_xgb.csv')
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])

    # Get the very latest actual level for comparison
    current_level = load_data()['water_level_m'].iloc[-1]
    pred_df['change_from_current'] = pred_df['water_level_m'] - current_level
    pred_df['trend'] = pred_df['change_from_current'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')

    def get_status(level):
        if level < 5: return 'Safe'
        elif 5 <= level < 10: return 'Semi-Critical'
        else: return 'Critical'

    pred_df['status'] = pred_df['water_level_m'].apply(get_status)
    return pred_df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- EARLY WARNING ALERTS ---
def check_critical_alerts(data):
    latest_record = data.iloc[-1]
    alerts = []
    if latest_record['water_level_m'] > 10:
        alerts.append({'message': f"🚨 CRITICAL: Water level at {latest_record['water_level_m']:.2f}m depth", 'color': 'red'})
    elif latest_record['water_level_m'] > 7:
        alerts.append({'message': f"⚠️ WARNING: Water level at {latest_record['water_level_m']:.2f}m depth", 'color': 'orange'})
    else:
        alerts.append({'message': f"✅ SAFE: Water level at {latest_record['water_level_m']:.2f}m depth", 'color': 'green'})
    return alerts

alerts = check_critical_alerts(df)
if alerts:
    st.sidebar.header("🚨 Early Warning Alerts")
    for alert in alerts:
        if alert['color'] == 'red': st.sidebar.error(alert['message'])
        elif alert['color'] == 'orange': st.sidebar.warning(alert['message'])
        else: st.sidebar.success(alert['message'])

# --- SIDEBAR FILTERS ---
st.sidebar.header("🎛️ Dashboard Controls")
view_mode = st.sidebar.radio("Select View Mode", ["Current Monitoring", "Future Predictions", "Research & Data Access"])

# --- MAIN DASHBOARD logic ---
if view_mode == "Current Monitoring":
    st.title("🌊 Borivali Groundwater Monitoring")
    
    # Timeframe and Season Filters
    timeframe = st.sidebar.selectbox("Select Time Range", ["All Time", "Last 7 Days", "Last 1 Month", "Last 3 Months", "Last 1 Year"])
    max_date = df['timestamp'].max()
    if timeframe == "Last 7 Days": filtered_df = df[df['timestamp'] >= max_date - timedelta(days=7)]
    elif timeframe == "Last 1 Month": filtered_df = df[df['timestamp'] >= max_date - timedelta(days=30)]
    else: filtered_df = df.copy()

    # KPI ROW
    current_val = filtered_df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Water Level", f"{current_val['water_level_m']:.2f} m")
    col2.metric("Period Rainfall", f"{filtered_df['rainfall_mm'].sum():.1f} mm")
    col3.metric("Total Extraction", f"{filtered_df['extraction_mcm'].sum():.2f} MCM")
    col4.metric("Current Status", current_val['status'])

    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "🗺️ Geo-Spatial", "⚖️ Comparison"])
    with tab1:
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(x=filtered_df['timestamp'], y=filtered_df['rainfall_mm'], name="Rainfall (mm)"))
        fig_rain.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['water_level_m'], name="Water Level (m)", yaxis="y2"))
        fig_rain.update_layout(yaxis2=dict(overlaying="y", side="right", autorange="reversed"), hovermode="x unified")
        st.plotly_chart(fig_rain, use_container_width=True)

elif view_mode == "Future Predictions":
    st.title("🔮 Borivali Future Groundwater Predictions")
    st.markdown("### Next 24-Hour Groundwater Level Forecast")

    if st.button("🚀 Generate Future Predictions", type="primary", use_container_width=True):
        with st.spinner("Executing Prediction Pipeline and XGBoost Model..."):
            try:
                # STEP 1: Execute pipeline first
                run_full_pipeline(uploaded_file)
                
                # STEP 2: Load predictions only after pipeline completes
                pred_df = load_predictions()
                
                st.success("✅ Predictions generated and loaded successfully!")

                # --- KPI Comparison ---
                col1, col2, col3, col4 = st.columns(4)
                current_val = df.iloc[-1]
                next_pred = pred_df.iloc[0]

                col1.metric("Current Level", f"{current_val['water_level_m']:.2f} m")
                col2.metric("Next Hour Prediction", f"{next_pred['water_level_m']:.2f} m", 
                           delta=f"{next_pred['change_from_current']:+.3f} m", delta_color="inverse")
                col3.metric("24-Hour Trend", f"{pred_df['water_level_m'].iloc[-1]:.2f} m", 
                           delta=f"{pred_df['water_level_m'].iloc[-1] - current_val['water_level_m']:+.3f} m", delta_color="inverse")
                col4.metric("Prediction Status", next_pred['status'])

                # --- Visualizations ---
                st.subheader("📊 24-Hour Forecast Analysis")
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=[pred_df['timestamp'].min(), pred_df['timestamp'].max()], 
                                            y=[current_val['water_level_m'], current_val['water_level_m']], 
                                            name='Current Level', line=dict(color='red', dash='dash')))
                fig_pred.add_trace(go.Scatter(x=pred_df['timestamp'], y=pred_df['water_level_m'], 
                                            mode='lines+markers', name='Predicted Level', line=dict(color='blue')))
                fig_pred.update_layout(yaxis_autorange="reversed", hovermode="x unified")
                st.plotly_chart(fig_pred, use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    fig_change = px.bar(pred_df, x='timestamp', y='change_from_current', color='trend',
                                        color_discrete_map={'Increase': 'red', 'Decrease': 'green'}, title="Hourly Change (m)")
                    st.plotly_chart(fig_change, use_container_width=True)
                with col_b:
                    status_counts = pred_df['status'].value_counts()
                    fig_status = px.pie(values=status_counts.values, names=status_counts.index, title="Risk Distribution")
                    st.plotly_chart(fig_status, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during the prediction cycle: {e}")
    else:
        st.info("Click the button above to run the analysis. This will process the latest DWLR data and generate a 24-hour outlook.")

elif view_mode == "Research & Data Access":
    st.title("🔬 Research & Data Access Portal")
    # ... [Rest of your Export and Research code remains the same] ...
    st.dataframe(df.head(10)) # Placeholder for your export logic