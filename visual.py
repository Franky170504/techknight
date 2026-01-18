import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime, timedelta
import subprocess
import sys

# --- PAGE CONFIG ---
st.set_page_config(page_title="Borivali Groundwater Analytics", layout="wide")

# Try to import prediction module (optional - for future predictions)
try:
    from src.prediction import run_full_pipeline
    PREDICTION_AVAILABLE = True
except ImportError as e:
    PREDICTION_AVAILABLE = False
    st.sidebar.warning("⚠️ Prediction features unavailable. Install sklearn and xgboost for full functionality.")

# File paths
DATA_FILE = "BORIVALI_DWLR_REALTIME.csv"
MULTI_DATA_FILE = "BORIVALI_DWLR_REALTIME_multi.csv"

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Try to load multi-station file first, fallback to single station
    try:
        df = pd.read_csv(MULTI_DATA_FILE)
        st.info(f"📊 Loaded multi-station data from {MULTI_DATA_FILE}")
    except FileNotFoundError:
        try:
            df = pd.read_csv(DATA_FILE)
            st.info(f"📊 Loaded single-station data from {DATA_FILE}")
        except FileNotFoundError:
            st.error(f"❌ No data files found. Expected {MULTI_DATA_FILE} or {DATA_FILE}")
            return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # SAFETY: If extraction_mcm is missing, create it based on demand
    if 'extraction_mcm' not in df.columns and 'demand_mcm' in df.columns:
        df['extraction_mcm'] = df['demand_mcm'] * 0.85 # Assuming 85% of demand is extracted
    elif 'extraction_mcm' not in df.columns:
        df['extraction_mcm'] = 0 # Fallback

    # Define Seasons
    def get_season(month):
        if month in [6, 7, 8, 9]: return 'Monsoon'
        if month in [10, 11]: return 'Post-Monsoon'
        if month in [12, 1, 2]: return 'Winter'
        return 'Summer'

    df['season'] = df['month'].apply(get_season)

    # Define Risk Status
    def get_status(level):
        if level < 5: return 'Safe'
        elif 5 <= level < 10: return 'Semi-Critical'
        else: return 'Critical'

    df['status'] = df['water_level_m'].apply(get_status)
    return df

@st.cache_data
def load_predictions():
    # Load future predictions data
    pred_df = pd.read_csv('notebooks/24hrs_xgb.csv')
    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])

    # Calculate change from current level
    current_level = load_data()['water_level_m'].iloc[-1]
    pred_df['change_from_current'] = pred_df['water_level_m'] - current_level
    pred_df['trend'] = pred_df['change_from_current'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')

    # Define Risk Status for predictions
    def get_status(level):
        if level < 5: 
            return 'Safe'
        elif 5 <= level < 10: 
            return 'Semi-Critical'
        else:
            return 'Critical'

    pred_df['status'] = pred_df['water_level_m'].apply(get_status)
    return pred_df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- EARLY WARNING ALERTS ---
def check_critical_alerts(data):
    """Check for critical groundwater level alerts"""
    latest_record = data.iloc[-1]
    alerts = []

    # Critical level alert (>10m depth = very low water)
    if latest_record['water_level_m'] > 10:
        alerts.append({
            'type': 'critical',
            'message': f"🚨 CRITICAL: Water level at {latest_record['water_level_m']:.2f}m depth - Immediate action required!",
            'color': 'red'
        })
    # Semi-critical alert (7-10m depth)
    elif latest_record['water_level_m'] > 7:
        alerts.append({
            'type': 'warning',
            'message': f"⚠️ WARNING: Water level at {latest_record['water_level_m']:.2f}m depth - Monitor closely",
            'color': 'orange'
        })
    # Safe level
    else:
        alerts.append({
            'type': 'safe',
            'message': f"✅ SAFE: Water level at {latest_record['water_level_m']:.2f}m depth - Normal conditions",
            'color': 'green'
        })

    # Check for rapid decline (compared to last 24 hours)
    if len(data) > 24:
        last_24h_avg = data.tail(24)['water_level_m'].mean()
        current_level = latest_record['water_level_m']
        change_24h = current_level - last_24h_avg

        if change_24h > 0.5:  # Rapid increase in depth = rapid decrease in water
            alerts.append({
                'type': 'rapid_change',
                'message': f"📉 RAPID DECLINE: Water level dropped {change_24h:.2f}m in last 24 hours",
                'color': 'red'
            })

    return alerts

# Display alerts
alerts = check_critical_alerts(df)
if alerts:
    st.sidebar.header("🚨 Early Warning Alerts")
    for alert in alerts:
        if alert['color'] == 'red':
            st.sidebar.error(alert['message'])
        elif alert['color'] == 'orange':
            st.sidebar.warning(alert['message'])
        else:
            st.sidebar.success(alert['message'])

# --- SIDEBAR FILTERS ---
st.sidebar.header("🎛️ Dashboard Controls")

# 0. View Mode Selector
view_mode = st.sidebar.radio(
    "Select View Mode",
    ["Current Monitoring", "Future Predictions", "Research & Data Access"],
    help="Switch between current data monitoring, future groundwater predictions, and research data access"
)

if view_mode == "Current Monitoring":
    # 1. Timeframe Selector
    timeframe = st.sidebar.selectbox(
        "Select Time Range",
        ["All Time", "Last 7 Days", "Last 1 Month", "Last 3 Months", "Last 1 Year"]
    )

    # 2. Station Selector
    available_stations = sorted(df['station_id'].unique())
    station_options = ["All Stations"] + available_stations
    selected_station = st.sidebar.selectbox(
        "Select GWRL Station",
        options=station_options,
        help="Choose which Ground Water Recording Location to analyze (or select 'All Stations' to view all)"
    )

    # Filter data based on timeframe
    max_date = df['timestamp'].max()
    if timeframe == "Last 7 Days":
        filtered_df = df[df['timestamp'] >= max_date - timedelta(days=7)]
    elif timeframe == "Last 1 Month":
        filtered_df = df[df['timestamp'] >= max_date - timedelta(days=30)]
    elif timeframe == "Last 3 Months":
        filtered_df = df[df['timestamp'] >= max_date - timedelta(days=90)]
    elif timeframe == "Last 1 Year":
        filtered_df = df[df['timestamp'] >= max_date - timedelta(days=365)]
    else:
        filtered_df = df.copy()

    # Filter by selected station (only if not "All Stations")
    if selected_station != "All Stations":
        filtered_df = filtered_df[filtered_df['station_id'] == selected_station]

    # 3. Season Selector
    seasons_available = df['season'].unique().tolist()
    selected_season = st.sidebar.multiselect("Filter by Season", options=seasons_available, default=seasons_available)
    filtered_df = filtered_df[filtered_df['season'].isin(selected_season)]

else:
    # For future predictions, use full dataset
    filtered_df = df.copy()

# --- MAIN DASHBOARD ---
if view_mode == "Current Monitoring":
    # Header with refresh button in corner
    col_title, col_refresh = st.columns([4, 1])

    with col_title:
        st.title("🌊 Borivali Groundwater Monitoring")
        if selected_station == "All Stations":
            st.markdown("### Real-time DWLR Analytics & Decision Support System - **All Stations Overview**")
        else:
            st.markdown(f"### Real-time DWLR Analytics & Decision Support System - **{selected_station}**")

    with col_refresh:
        # Corner refresh button
        if st.button("🔄 Refresh", help="Update groundwater data and refresh dashboard"):
            with st.spinner("Updating groundwater data... This may take a few moments."):
                try:
                    # Run the groundwater script
                    result = subprocess.run([sys.executable, "groundwater1.py"],
                                          capture_output=True, text=True, cwd=".")

                    if result.returncode == 0:
                        st.success("✅ Data refreshed successfully!")
                        # Clear cache and rerun the app
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"❌ Error updating data: {result.stderr}")

                except Exception as e:
                    st.error(f"❌ Failed to refresh data: {str(e)}")

    if filtered_df.empty:
        st.warning("No data found for the selected filters.")
        st.stop()

    # KPI ROW
    col1, col2, col3, col4 = st.columns(4)

    if selected_station == "All Stations":
        # Aggregate KPIs across all stations
        latest_data = filtered_df.groupby('station_id').tail(1)  # Get latest record for each station
        avg_water_level = latest_data['water_level_m'].mean()
        total_rainfall = filtered_df['rainfall_mm'].sum()
        total_extraction = filtered_df['extraction_mcm'].sum()

        # Most common status across stations
        status_counts = latest_data['status'].value_counts()
        most_common_status = status_counts.index[0]

        col1.metric("Avg Current Water Level", f"{avg_water_level:.2f} m")
        col2.metric("Total Period Rainfall", f"{total_rainfall:.1f} mm")
        col3.metric("Total Extraction", f"{total_extraction:.2f} MCM")
        col4.metric("Most Common Status", most_common_status)

        # Add station count info
        st.info(f"📊 **Analyzing {len(available_stations)} stations** - Showing aggregated data across all locations")
    else:
        current_val = filtered_df.iloc[-1]

        col1.metric("Current Water Level", f"{current_val['water_level_m']:.2f} m")
        col2.metric("Period Rainfall", f"{filtered_df['rainfall_mm'].sum():.1f} mm")
        col3.metric("Total Extraction", f"{filtered_df['extraction_mcm'].sum():.2f} MCM")
        col4.metric("Current Status", current_val['status'])

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Trend Analysis", "🗺️ Geo-Spatial", "⚖️ Comparison"])

    with tab1:
        if selected_station == "All Stations":
            st.subheader("Groundwater & Rainfall Trends - All Stations Overview")
        else:
            st.subheader(f"Groundwater & Rainfall Trends - {selected_station}")

        if selected_station == "All Stations":
            # For all stations, show average trends
            avg_data = filtered_df.groupby('timestamp').agg({
                'water_level_m': 'mean',
                'rainfall_mm': 'mean',
                'temperature_c': 'mean'
            }).reset_index()

            # Rainfall vs Water Level (Dual Axis) - Average across all stations
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(x=avg_data['timestamp'], y=avg_data['rainfall_mm'], name="Avg Rainfall (mm)", marker_color='rgba(0, 123, 255, 0.3)'))
            fig_rain.add_trace(go.Scatter(x=avg_data['timestamp'], y=avg_data['water_level_m'], name="Avg Water Level (m)", yaxis="y2", line=dict(color='red', width=2)))

            fig_rain.update_layout(
                title="Average Rainfall vs. Water Level Depth Across All Stations (Lower Water Level = More Water)",
                yaxis=dict(title="Rainfall (mm)"),
                yaxis2=dict(title="Water Level Depth (m)", overlaying="y", side="right", autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig_rain, use_container_width=True)

            # Temperature Trend - Average across all stations
            fig_temp = px.line(avg_data, x='timestamp', y='temperature_c', title="Average Temperature Cycle Across All Stations (°C)", color_discrete_sequence=['orange'])
            st.plotly_chart(fig_temp, use_container_width=True)

            # Station-wise comparison chart
            st.subheader("📊 Station-wise Water Level Comparison")
            fig_comparison = px.line(filtered_df, x='timestamp', y='water_level_m', color='station_id',
                                   title="Water Level Trends by Station")
            st.plotly_chart(fig_comparison, use_container_width=True)

        else:
            # Rainfall vs Water Level (Dual Axis)
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(x=filtered_df['timestamp'], y=filtered_df['rainfall_mm'], name="Rainfall (mm)", marker_color='rgba(0, 123, 255, 0.3)'))
            fig_rain.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['water_level_m'], name="Water Level (m)", yaxis="y2", line=dict(color='red', width=2)))

            fig_rain.update_layout(
                title="Rainfall vs. Water Level Depth (Note: Lower Water Level = More Water)",
                yaxis=dict(title="Rainfall (mm)"),
                yaxis2=dict(title="Water Level Depth (m)", overlaying="y", side="right", autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig_rain, use_container_width=True)

            # Temperature Trend
            fig_temp = px.line(filtered_df, x='timestamp', y='temperature_c', title="Temperature Cycle (°C)", color_discrete_sequence=['orange'])
            st.plotly_chart(fig_temp, use_container_width=True)

    with tab2:
        if selected_station == "All Stations":
            st.subheader("Station Locations & Current Risk - All Stations Overview")

            # Get latest record for each station from filtered data (includes time filtering)
            latest_records = (
                filtered_df
                .sort_values('timestamp')
                .groupby('station_id')
                .tail(1)
                .copy()
            )

            # Debug information
            with st.expander("🔍 Map Data Debug Info"):
                st.write("Latest records for all stations:")
                st.dataframe(latest_records[['station_id', 'status', 'water_level_m', 'latitude', 'longitude', 'timestamp']], use_container_width=True)

        else:
            st.subheader(f"Station Location & Current Risk - {selected_station}")

            # Latest record for map
            map_record = filtered_df.tail(1).copy()

            # Debug information
            with st.expander("🔍 Map Data Debug Info"):
                st.write(f"Station: {selected_station}")
                st.write(f"Status: {map_record['status'].iloc[0]}")
                st.write(f"Water Level: {map_record['water_level_m'].iloc[0]}m")
                st.write(f"Latitude: {map_record['latitude'].iloc[0]}")
                st.write(f"Longitude: {map_record['longitude'].iloc[0]}")
                st.write(f"Aquifer Area: {map_record['aquifer_area_sqkm'].iloc[0]} sq km")
                st.write(f"Specific Yield: {map_record['specific_yield'].iloc[0]}")
                st.write(f"Land Use: {map_record['land_use_type'].iloc[0]}")

        # Enhanced Map Colors with better visibility
        def get_map_color(status):
            """Return RGBA color values for map markers"""
            if status == 'Safe':
                return [34, 139, 34, 200]  # Dark green
            elif status == 'Semi-Critical':
                return [255, 140, 0, 200]  # Dark orange
            elif status == 'Critical':
                return [220, 20, 60, 200]  # Crimson red
            else:
                return [128, 128, 128, 200]  # Gray for unknown

        # Handle both single station and all stations display
        if selected_station == "All Stations":
            # Apply color mapping for all stations
            latest_records['color'] = latest_records['status'].apply(get_map_color)

            # Create map with enhanced settings for all stations
            try:
                # Calculate center point for all stations
                center_lat = latest_records['latitude'].mean()
                center_lon = latest_records['longitude'].mean()

                deck = pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=float(center_lat),
                        longitude=float(center_lon),
                        zoom=13,  # Slightly zoomed out to show all stations
                        pitch=45,
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=latest_records,
                            get_position=['longitude', 'latitude'],
                            get_color='color',
                            get_radius=600,  # Larger radius for better visibility when multiple stations
                            pickable=True,
                            stroked=True,
                            filled=True,
                            line_width_min_pixels=2,
                            get_line_color=[255, 255, 255, 255],  # White border
                        ),
                    ],
                    tooltip={
                        "html": """
                        <b>Station:</b> {station_id}<br/>
                        <b>Water Level:</b> {water_level_m} m<br/>
                        <b>Status:</b> {status}<br/>
                        <b>Rainfall:</b> {rainfall_mm} mm<br/>
                        <b>Temperature:</b> {temperature_c}°C<br/>
                        <b>Last Updated:</b> {timestamp}
                        """,
                        "style": {"color": "white"}
                    }
                )

                st.pydeck_chart(deck)

                # Status summary for all stations
                station_summary = latest_records.groupby('status').size().to_dict()
                summary_text = " | ".join([f"{status}: {count}" for status, count in station_summary.items()])
                st.info(f"**All Stations Overview:** {summary_text}")

                # Individual station status indicators
                st.subheader("📍 Individual Station Status")
                for _, station in latest_records.iterrows():
                    status_color = {
                        'Safe': '🟢',
                        'Semi-Critical': '🟠',
                        'Critical': '🔴'
                    }.get(station['status'], '⚪')
                    st.write(f"{status_color} **{station['station_id']}:** {station['status']} - Water Level: {station['water_level_m']:.2f}m")

            except Exception as e:
                st.error(f"Map loading error: {e}")
                # Fallback: show coordinates for all stations
                st.write("📍 All Station Coordinates:")
                st.dataframe(latest_records[['station_id', 'latitude', 'longitude', 'status', 'water_level_m']], use_container_width=True)

        else:
            # Apply color mapping for single station
            map_record['color'] = map_record['status'].apply(get_map_color)

            # Create map with enhanced settings for single station
            try:
                deck = pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=float(map_record['latitude'].iloc[0]),
                        longitude=float(map_record['longitude'].iloc[0]),
                        zoom=15,
                        pitch=45,
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=map_record,
                            get_position=['longitude', 'latitude'],
                            get_color='color',
                            get_radius=500,  # Increased radius for better visibility
                            pickable=True,
                            stroked=True,
                            filled=True,
                            line_width_min_pixels=2,
                            get_line_color=[255, 255, 255, 255],  # White border
                        ),
                    ],
                    tooltip={
                        "html": """
                        <b>Station:</b> {station_id}<br/>
                        <b>Water Level:</b> {water_level_m} m<br/>
                        <b>Status:</b> {status}<br/>
                        <b>Rainfall:</b> {rainfall_mm} mm<br/>
                        <b>Temperature:</b> {temperature_c}°C
                        """,
                        "style": {"color": "white"}
                    }
                )

                st.pydeck_chart(deck)

                # Status indicator below map
                status_color = {
                    'Safe': '🟢',
                    'Semi-Critical': '🟠',
                    'Critical': '🔴'
                }.get(map_record['status'].iloc[0], '⚪')

                st.info(f"{status_color} **{selected_station} Status:** {map_record['status'].iloc[0]} - Water Level: {map_record['water_level_m'].iloc[0]:.2f}m")

            except Exception as e:
                st.error(f"Map loading error: {e}")
                # Fallback: show coordinates
                st.write("📍 Station Coordinates:")
                st.write(f"Latitude: {map_record['latitude'].iloc[0]}")
                st.write(f"Longitude: {map_record['longitude'].iloc[0]}")

    with tab3:
        if selected_station == "All Stations":
            st.subheader("Seasonal Stress - All Stations Overview")

            # Group by season and station for all stations analysis
            seas_station_avg = df.groupby(['season', 'station_id'])['water_level_m'].mean().reset_index()
            fig_s = px.bar(seas_station_avg, x='season', y='water_level_m', color='station_id',
                          title="Average Depth by Season for All Stations", barmode='group')
            st.plotly_chart(fig_s, use_container_width=True)

            # Add station comparison
            st.subheader("📊 Station-wise Comparison")
            station_comparison = df.groupby('station_id')['water_level_m'].agg(['mean', 'min', 'max']).round(2).reset_index()
            station_comparison.columns = ['Station', 'Avg Depth', 'Min Depth', 'Max Depth']
            st.dataframe(station_comparison, use_container_width=True)

        else:
            st.subheader(f"Seasonal Stress - {selected_station}")
            seas_avg = df[df['station_id'] == selected_station].groupby('season')['water_level_m'].mean().reset_index()
            fig_s = px.bar(seas_avg, x='season', y='water_level_m', color='season', title="Average Depth by Season")
            st.plotly_chart(fig_s, use_container_width=True)

        st.divider()
        st.info("💡 **Judge Tip:** Point out the inverse relationship in the top graph—when the blue bars (rain) are high, the red line (depth) should trend downwards as the aquifer recharges.")

elif view_mode == "Future Predictions":
    st.title("🔮 Borivali Future Groundwater Predictions")
    st.markdown("### Next 24-Hour Groundwater Level Forecast")

    # Initialize predictions data variable
    pred_df = None
    predictions_loaded = False

    # Show prediction trigger button
    if PREDICTION_AVAILABLE:
        if st.button("🚀 Generate Future Predictions", type="primary", use_container_width=True):
            with st.spinner("Generating predictions... This may take a few moments."):
                try:
                    # First, run the prediction pipeline
                    prediction_results = run_full_pipeline(MULTI_DATA_FILE)

                    # Save the prediction results to CSV file
                    prediction_results.to_csv('notebooks/24hrs_xgb.csv', index=False)
                    st.success("✅ Prediction pipeline executed and results saved!")

                    # Clear prediction cache to ensure fresh data is loaded
                    st.cache_data.clear()

                    # Then, load the newly generated predictions
                    pred_df = load_predictions()
                    predictions_loaded = True
                    st.success("✅ Predictions data loaded successfully!")

                except Exception as e:
                    st.error(f"❌ Error in prediction pipeline: {str(e)}")
                    st.stop()
    else:
        st.warning("⚠️ Prediction functionality requires sklearn and xgboost packages.")
        st.info("Install with: `pip install scikit-learn xgboost`")
        if st.button("🚀 Generate Future Predictions", type="primary", use_container_width=True, disabled=True):
            pass

    # Only show predictions if they have been loaded
    if predictions_loaded and pred_df is not None:
        # Current vs Predicted KPI Comparison
        col1, col2, col3, col4 = st.columns(4)
        current_val = df.iloc[-1]
        next_pred = pred_df.iloc[0]

        col1.metric(
            "Current Level",
            f"{current_val['water_level_m']:.2f} m",
            delta=None
        )
        col2.metric(
            "Next Hour Prediction",
            f"{next_pred['water_level_m']:.2f} m",
            delta=f"{next_pred['change_from_current']:+.3f} m",
            delta_color="inverse"  # Red for increase (bad), green for decrease (good)
        )
        col3.metric(
            "24-Hour Trend",
            f"{pred_df['water_level_m'].iloc[-1]:.2f} m",
            delta=f"{pred_df['water_level_m'].iloc[-1] - current_val['water_level_m']:+.3f} m",
            delta_color="inverse"
        )
        col4.metric("Prediction Status", next_pred['status'])

        # --- PREDICTIONS VISUALIZATION ---
        st.subheader("📊 24-Hour Forecast Analysis")

        # Main prediction trend chart
        fig_pred = go.Figure()

        # Add current level reference line
        fig_pred.add_trace(go.Scatter(
            x=[pred_df['timestamp'].min(), pred_df['timestamp'].max()],
            y=[current_val['water_level_m'], current_val['water_level_m']],
            mode='lines',
            name='Current Level',
            line=dict(color='red', dash='dash', width=2),
            showlegend=True
        ))

        # Add predicted levels
        fig_pred.add_trace(go.Scatter(
            x=pred_df['timestamp'],
            y=pred_df['water_level_m'],
            mode='lines+markers',
            name='Predicted Level',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color=pred_df['change_from_current'].apply(lambda x: 'green' if x < 0 else 'red')),
            showlegend=True
        ))

        fig_pred.update_layout(
            title="Groundwater Level Forecast (Next 24 Hours)",
            xaxis_title="Time",
            yaxis_title="Water Level (m)",
            yaxis_autorange="reversed",  # Lower values = more water
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        # Change analysis chart
        st.subheader("📈 Hourly Changes & Trends")

        col_a, col_b = st.columns(2)

        with col_a:
            # Change from current level
            fig_change = px.bar(
                pred_df,
                x='timestamp',
                y='change_from_current',
                title="Change from Current Level (m)",
                color='trend',
                color_discrete_map={'Increase': 'red', 'Decrease': 'green'},
                labels={'change_from_current': 'Change (m)', 'timestamp': 'Time'}
            )
            fig_change.update_layout(showlegend=False)
            st.plotly_chart(fig_change, use_container_width=True)

        with col_b:
            # Risk status over time
            status_counts = pred_df['status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Predicted Risk Distribution (24h)",
                color=status_counts.index,
                color_discrete_map={'Safe': 'green', 'Semi-Critical': 'orange', 'Critical': 'red'}
            )
            st.plotly_chart(fig_status, use_container_width=True)

        # Temperature and rainfall predictions
        st.subheader("🌡️ Supporting Factors Forecast")

        col_c, col_d = st.columns(2)

        with col_c:
            fig_temp_pred = px.line(
                pred_df,
                x='timestamp',
                y='temperature_c',
                title="Predicted Temperature (°C)",
                color_discrete_sequence=['orange']
            )
            st.plotly_chart(fig_temp_pred, use_container_width=True)

        with col_d:
            fig_rain_pred = px.bar(
                pred_df,
                x='timestamp',
                y='rainfall_mm',
                title="Predicted Rainfall (mm)",
                color_discrete_sequence=['blue']
            )
            st.plotly_chart(fig_rain_pred, use_container_width=True)

        st.divider()
        st.info("💡 **Prediction Insights:** Green markers indicate decreasing water levels (improving conditions), red markers show increasing levels (worsening conditions). Lower water level depths mean more groundwater available.")

        # Predicted values table
        st.subheader("📋 Detailed Forecast Values")
        st.markdown("### 24-Hour Prediction Table")

        # Format the prediction data for display
        display_df = pred_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df[['timestamp', 'water_level_m', 'rainfall_mm', 'temperature_c', 'change_from_current', 'status']]

        # Rename columns for better display
        display_df.columns = ['Time', 'Water Level (m)', 'Rainfall (mm)', 'Temperature (°C)', 'Change from Current (m)', 'Risk Status']

        # Format numerical columns
        display_df['Water Level (m)'] = display_df['Water Level (m)'].round(3)
        display_df['Rainfall (mm)'] = display_df['Rainfall (mm)'].round(2)
        display_df['Temperature (°C)'] = display_df['Temperature (°C)'].round(1)
        display_df['Change from Current (m)'] = display_df['Change from Current (m)'].round(3)

        st.dataframe(display_df, use_container_width=True)

        # Add summary statistics
        st.subheader("📊 Prediction Summary")
        col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)

        with col_summary1:
            st.metric("Average Water Level", f"{pred_df['water_level_m'].mean():.2f} m")
        with col_summary2:
            st.metric("Total Predicted Rainfall", f"{pred_df['rainfall_mm'].sum():.1f} mm")
        with col_summary3:
            st.metric("Max Water Level Change", f"{pred_df['change_from_current'].max():+.3f} m")
        with col_summary4:
            most_common_status = pred_df['status'].mode().iloc[0]
            st.metric("Most Common Status", most_common_status)

elif view_mode == "Research & Data Access":
    st.title("🔬 Research & Data Access Portal")
    st.markdown("### Historical & Real-time Groundwater Data for Analysis & Research")

    # --- DATA EXPORT CONTROLS ---
    st.header("📊 Data Export & Filtering")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Date Range Selection")
        # Date range picker
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()

        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

        # Hour range selector
        st.subheader("🕐 Hour Range Selection")
        start_hour = st.slider("Start Hour (0-23)", 0, 23, 0)
        end_hour = st.slider("End Hour (0-23)", 0, 23, 23)

    with col2:
        st.subheader("📋 Data Filters")

        # Column selection
        all_columns = df.columns.tolist()
        default_columns = ['timestamp', 'water_level_m', 'rainfall_mm', 'temperature_c', 'extraction_mcm', 'season', 'status']
        selected_columns = st.multiselect(
            "Select Columns to Export",
            options=all_columns,
            default=default_columns,
            help="Choose which data columns to include in your export"
        )

        # Additional filters
        export_season = st.multiselect("Filter by Season", options=df['season'].unique().tolist(), default=df['season'].unique().tolist())
        export_status = st.multiselect("Filter by Status", options=df['status'].unique().tolist(), default=df['status'].unique().tolist())

    # --- FILTER AND PREVIEW DATA ---
    # Apply filters
    filtered_export_df = df.copy()

    # Date and hour filtering
    filtered_export_df['date'] = filtered_export_df['timestamp'].dt.date
    filtered_export_df['hour'] = filtered_export_df['timestamp'].dt.hour

    filtered_export_df = filtered_export_df[
        (filtered_export_df['date'] >= start_date) &
        (filtered_export_df['date'] <= end_date) &
        (filtered_export_df['hour'] >= start_hour) &
        (filtered_export_df['hour'] <= end_hour)
    ]

    # Season and status filtering
    filtered_export_df = filtered_export_df[
        (filtered_export_df['season'].isin(export_season)) &
        (filtered_export_df['status'].isin(export_status))
    ]

    # Select only requested columns
    if selected_columns:
        # Always include timestamp
        columns_to_export = ['timestamp'] + [col for col in selected_columns if col != 'timestamp']
        filtered_export_df = filtered_export_df[columns_to_export]

    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(filtered_export_df.head(100), use_container_width=True)

    # Data statistics
    st.subheader("📈 Export Summary")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Records", f"{len(filtered_export_df):,}")
    col_b.metric("Date Range", f"{start_date} to {end_date}")
    col_c.metric("Hour Range", f"{start_hour}:00 - {end_hour}:00")
    col_d.metric("Columns", len(selected_columns))

    # --- EXPORT OPTIONS ---
    st.header("💾 Export Data")

    # CSV Export
    if st.button("📄 Download as CSV", type="primary", use_container_width=True):
        csv_data = filtered_export_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Click to Download CSV",
            data=csv_data,
            file_name=f"groundwater_data_{start_date}_to_{end_date}.csv",
            mime="text/csv",
            key="csv_download"
        )
        st.success("✅ CSV file ready for download!")

    # Excel Export (requires openpyxl)
    try:
        import io

        if st.button("📊 Download as Excel", type="primary", use_container_width=True):
            # Create Excel file in memory
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_export_df.to_excel(writer, sheet_name='Groundwater_Data', index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets['Groundwater_Data']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            buffer.seek(0)
            st.download_button(
                label="⬇️ Click to Download Excel",
                data=buffer,
                file_name=f"groundwater_data_{start_date}_to_{end_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )
            st.success("✅ Excel file ready for download!")

    except ImportError:
        st.warning("⚠️ Excel export requires 'openpyxl' package. Install with: pip install openpyxl")

    # --- RESEARCH TOOLS ---
    st.header("🔬 Research Tools & Analytics")

    # Basic statistics
    st.subheader("📊 Basic Statistics")
    if not filtered_export_df.empty and 'water_level_m' in filtered_export_df.columns:
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

        with stats_col1:
            st.metric("Mean Water Level", f"{filtered_export_df['water_level_m'].mean():.2f}m")
        with stats_col2:
            st.metric("Min Water Level", f"{filtered_export_df['water_level_m'].min():.2f}m")
        with stats_col3:
            st.metric("Max Water Level", f"{filtered_export_df['water_level_m'].max():.2f}m")
        with stats_col4:
            st.metric("Std Deviation", f"{filtered_export_df['water_level_m'].std():.2f}m")

    # Data quality information
    st.subheader("📋 Data Quality Information")
    quality_info = pd.DataFrame({
        'Metric': ['Total Records', 'Missing Values', 'Duplicate Records', 'Date Range'],
        'Value': [
            len(filtered_export_df),
            filtered_export_df.isnull().sum().sum(),
            filtered_export_df.duplicated().sum(),
            f"{filtered_export_df['timestamp'].min()} to {filtered_export_df['timestamp'].max()}"
        ]
    })
    st.table(quality_info)

    # Usage guidelines
    st.header("📚 Usage Guidelines & Citation")
    st.info("""
    **Research Data Usage Guidelines:**

    1. **Citation**: When using this data in publications, please cite: "Borivali Groundwater Monitoring Dataset, Maharashtra, India"

    2. **Data Quality**: Data has been quality-checked but users should perform their own validation for research purposes.

    3. **Units**: Water levels in meters (depth), rainfall in mm, temperature in °C, extraction in million cubic meters (MCM).

    4. **Temporal Resolution**: Hourly data with occasional gaps due to sensor maintenance.

    5. **Contact**: For questions about data methodology or additional datasets, contact the groundwater monitoring team.
    """)

    st.divider()
    st.success("🎯 **Ready for Research**: Your filtered dataset is prepared for download and analysis!")