import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
from datetime import timedelta
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import time

# Set Streamlit page configuration
st.set_page_config(page_title="🚀 Fintech Time Machine", layout="wide")

# Custom CSS for Unified Professional Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Poppins&display=swap');
    .main { 
        background: linear-gradient(135deg, #0E0E1F 0%, #1C2526 100%);
        padding: 20px; 
        background-image: url('https://www.transparenttextures.com/patterns/gplay.png');
        background-size: cover;
    }
    .stSidebar { 
        background: linear-gradient(135deg, #1C2526, #2C3E50); 
        padding: 20px; 
        border-right: 2px solid #00FFE5; 
        box-shadow: 0 0 15px #00FFE5; 
    }
    h1, h2, h3, h4 { 
        font-family: 'Orbitron', sans-serif; 
        color: #00FFE5; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
        text-shadow: 0 0 10px #00FFE566; 
    }
    .stMarkdown, .stWrite, .stMetric, label, .stRadio > label, .stSelectbox > label { 
        font-family: 'Poppins', sans-serif; 
        color: #E0E0E0; 
    }
    .stButton>button { 
        background-color: #00BFFF; 
        color: #FFFFFF; 
        font-family: 'Orbitron', sans-serif; 
        font-size: 16px; 
        border-radius: 15px; 
        padding: 12px 24px; 
        border: none; 
        transition: all 0.3s ease; 
        box-shadow: 0 0 10px #00BFFF; 
        animation: pulse 2s infinite; 
    }
    .stButton>button:hover { 
        background-color: #FF007A; 
        color: #FFFFFF; 
        transform: scale(1.1); 
        box-shadow: 0 0 20px #FF007A; 
    }
    @keyframes pulse { 
        0% { box-shadow: 0 0 10px #00BFFF; } 
        50% { box-shadow: 0 0 20px #00BFFF; } 
        100% { box-shadow: 0 0 10px #00BFFF; } 
    }
    .stSuccess { 
        background-color: #1C2526; 
        color: #FFFFFF; 
        border: 2px solid #00FFE5; 
        border-radius: 10px; 
        padding: 10px; 
        animation: fadeIn 0.5s; 
        box-shadow: 0 0 15px #00FFE566; 
    }
    .stMetric { 
        background-color: #1C1F3A; 
        border-left: 4px solid #00FFE5; 
        border-radius: 10px; 
        padding: 10px; 
        box-shadow: 0 0 10px #00FFE566; 
        animation: glow 1.5s infinite; 
    }
    @keyframes glow { 
        0% { box-shadow: 0 0 5px #00FFE5; } 
        50% { box-shadow: 0 0 15px #00FFE5; } 
        100% { box-shadow: 0 0 5px #00FFE5; } 
    }
    .section { 
        opacity: 0; 
        animation: fadeIn 1s forwards; 
    }
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(20px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    .icon-placeholder { 
        display: inline-block; 
        width: 30px; 
        height: 30px; 
        background-color: #FF007A; 
        border-radius: 50%; 
        margin-right: 10px; 
        vertical-align: middle; 
        box-shadow: 0 0 10px #FF007A; 
    }
    .tooltip { 
        position: relative; 
        display: inline-block; 
    }
    .tooltip .tooltiptext { 
        visibility: hidden; 
        width: 120px; 
        background-color: #FF007A; 
        color: #FFFFFF; 
        text-align: center; 
        border-radius: 6px; 
        padding: 5px; 
        position: absolute; 
        z-index: 1; 
        bottom: 125%; 
        left: 50%; 
        margin-left: -60px; 
        opacity: 0; 
        transition: opacity 0.3s; 
        box-shadow: 0 0 10px #FF007A; 
    }
    .tooltip:hover .tooltiptext { 
        visibility: visible; 
        opacity: 1; 
    }
    .stSidebar .element-container { 
        margin-bottom: 15px; 
        display: grid; 
        grid-template-columns: auto 1fr; 
        align-items: center; 
    }
    .stSidebar .element-container .stRadio > label, .stSidebar .element-container .stSelectbox > label, .stSidebar .element-container .stFileUploader > label { 
        margin: 0; 
        padding: 5px; 
        background: #1C1F3A; 
        border-radius: 8px; 
        box-shadow: 0 0 5px #00FFE5; 
    }
    .stSidebar .element-container input, .stSidebar .element-container select, .stSlider > div { 
        background: #1C1F3A; 
        color: #FFFFFF; 
        border: 2px solid #00FFE5; 
        border-radius: 8px; 
    }
    .stSidebar .element-container input:focus { 
        border-color: #FF007A; 
        box-shadow: 0 0 10px #FF007A; 
    }
    .stError { 
        background-color: #1C2526; 
        color: #FFFFFF; 
        border: 2px solid #FF007A; 
        border-radius: 10px; 
        padding: 10px; 
        animation: shake 0.5s; 
    }
    @keyframes shake { 
        0%, 100% { transform: translateX(0); } 
        25% { transform: translateX(-5px); } 
        50% { transform: translateX(5px); } 
        75% { transform: translateX(-5px); } 
    }
    .stSelectbox > div > div { 
        background-color: #1C1F3A; 
        color: #FFFFFF; 
        border: 2px solid #00FFE5; 
        border-radius: 8px; 
    }
    .stSelectbox > div > div:hover { 
        border-color: #FFFF00; 
        box-shadow: 0 0 10px #FFFF00; 
    }
    ::-webkit-scrollbar { 
        width: 8px; 
    }
    ::-webkit-scrollbar-track { 
        background: #1C1F3A; 
    }
    ::-webkit-scrollbar-thumb { 
        background: #00FFE5; 
    }
</style>
""", unsafe_allow_html=True)

# Matrix Animation HTML with Neon Pink Letters
matrix_html = """ 
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      background: black;
      overflow: hidden;
    }
    canvas {
      display: block;
    }
  </style>
</head>
<body>
  <canvas id="matrixCanvas"></canvas>
  <script>
    const canvas = document.getElementById('matrixCanvas');
    const ctx = canvas.getContext('2d');
    canvas.height = window.innerHeight;
    canvas.width = window.innerWidth;
    const letters = 'アァイィウヴエェオカガキギクグケゲコゴサザシジスズセゼソゾタダチッヂヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモヤユヨラリルレロワヲンABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops = Array.from({ length: columns }).fill(1);
    function draw() {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#FF007A';
      ctx.font = fontSize + 'px monospace';
      for (let i = 0; i < drops.length; i++) {
        const text = letters.charAt(Math.floor(Math.random() * letters.length));
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    }
    setInterval(draw, 33);
  </script>
</body>
</html>
"""

# Initialize session state with safeguards
if "current_level" not in st.session_state or not isinstance(st.session_state.current_level, dict):
    st.session_state.current_level = {"galaxy": 1, "portfolio": 1}
if "fincoins" not in st.session_state or not isinstance(st.session_state.fincoins, (int, float)):
    st.session_state.fincoins = 0
if "galaxy_data" not in st.session_state:
    st.session_state.galaxy_data = None
if "portfolio_data" not in st.session_state:
    st.session_state.portfolio_data = None
if "galaxy_features" not in st.session_state:
    st.session_state.galaxy_features = None
if "cluster_guess" not in st.session_state or not isinstance(st.session_state.cluster_guess, (int, float)):
    st.session_state.cluster_guess = None
if "galaxy_model" not in st.session_state:
    st.session_state.galaxy_model = None
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "labels" not in st.session_state:
    st.session_state.labels = None
if "portfolio_model" not in st.session_state:
    st.session_state.portfolio_model = None
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

# Intro Animation
if not st.session_state.intro_shown:
    st.session_state.intro_shown = True
    placeholder = st.empty()
    with placeholder.container():
        components.html(matrix_html, height=600, width=800)
    time.sleep(4)
    placeholder.empty()

# Sidebar Scoreboard and Inputs
st.sidebar.markdown('<div class="section">', unsafe_allow_html=True)
st.sidebar.header("DOPE TIME MACHINE")
st.sidebar.metric("FinCoins", st.session_state.fincoins, delta_color="normal")
try:
    st.sidebar.write(f"Galaxy Level: {st.session_state.current_level['galaxy']}/7")
    st.sidebar.write(f"Portfolio Level: {st.session_state.current_level['portfolio']}/5")
except KeyError as e:
    st.sidebar.error(f"Error accessing level: {e}. Resetting levels.")
    st.session_state.current_level = {"galaxy": 1, "portfolio": 1}
if st.session_state.fincoins >= 100:
    st.sidebar.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZjlvdmF0MzY4ZWtiN2U3YnQxeHBhYWowMmc5eWRqY2p4cnh3bXVxayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bztUNMLxXzSaQ0SvHv/giphy.gif", caption="[ICON: Time Commander]", width=100)
elif st.session_state.fincoins >= 60:
    st.sidebar.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExenF5MmR3ZGl4NHhpY2doaHY5dnZhdHcwdG9ocWpwMWdoc2pxdGk5eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6ZteW16VNCpTpnWg/giphy.gif", caption="[ICON: Time Explorer]", width=100)

with st.sidebar.expander("🌠 Galaxy Mapper Controls", expanded=False):
    data_source = st.radio("Pick Star Map", ["Upload Kragle Dataset", "Fetch Yahoo Finance Data"], format_func=lambda x: f"🌍 {x}")
    fintech_tickers = ["PYPL", "SQ", "ADYEN.AS", "FI", "INTU"]
    galaxy_ticker = st.selectbox("Choose Fintech Planet", fintech_tickers, format_func=lambda x: f"🪐 {x}") if data_source == "Fetch Yahoo Finance Data" else None
    galaxy_uploaded_file = st.file_uploader("Drop Kragle CSV", type="csv", help="Upload your custom dataset") if data_source == "Upload Kragle Dataset" else None
    try:
        galaxy_start_date = st.date_input("Start Date (Galaxy)", datetime.datetime.now() - timedelta(days=365))
        galaxy_end_date = st.date_input("End Date (Galaxy)", datetime.datetime.now())
        if galaxy_start_date >= galaxy_end_date:
            st.sidebar.error("Start Date must be before End Date!")
            galaxy_start_date = datetime.datetime.now() - timedelta(days=365)
            galaxy_end_date = datetime.datetime.now()
    except Exception as e:
        st.sidebar.error(f"Date error: {e}. Using defaults.")
        galaxy_start_date = datetime.datetime.now() - timedelta(days=365)
        galaxy_end_date = datetime.datetime.now()

with st.sidebar.expander("📅 Portfolio Simulation Controls", expanded=False):
    portfolio_ticker = st.text_input("📈 Stock Symbol", "AAPL")
    try:
        portfolio_start_date = st.date_input("🕰️ Start Date (Portfolio)", datetime.datetime.now() - timedelta(days=365 * 10))
        if portfolio_start_date >= datetime.datetime.now().date():
            st.sidebar.error("Start Date must be in the past!")
            portfolio_start_date = datetime.datetime.now() - timedelta(days=365 * 10)
    except Exception as e:
        st.sidebar.error(f"Date error: {e}. Using default.")
        portfolio_start_date = datetime.datetime.now() - timedelta(days=365 * 10)
    investment_amount = st.number_input("💵 Amount Invested ($)", min_value=100, value=1000)
    future_days = st.slider("📅 Days into Future", 1, 180, 30)
    portfolio_uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"], help="Upload CSV with 'Date' and 'Close' columns")
    simulate = st.button("🚀 Simulate Portfolio")

st.sidebar.markdown("🧠 **Simulated Sentiment**:")
sentiments = ["🚀 Bullish", "😐 Neutral", "🐻 Bearish"]
sentiment = np.random.choice(sentiments, p=[0.4, 0.4, 0.2])
st.sidebar.success(f"News Sentiment: {sentiment}")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Page Navigation
st.markdown('<div class="section">', unsafe_allow_html=True)
st.title("🚀 FINTECH TIME MACHINE")
st.markdown("""
Welcome to the ultimate financial adventure! Map the **Fintech Galaxy** by clustering companies into factions, or jump into the **Time Traveler's Portfolio** to simulate and predict stock performance. Stack **FinCoins**, unlock dope ranks, and become the **Time Commander**!
""")
st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDJ0ODgxemtkZnNwYTRlbjhjc3hkMjJxODd6bTc4OHl2Y2Nmcjl4bSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6Ztdux6MvDwlnbck/giphy.gif", caption="Launch the Hustle!", width=400)
page = st.selectbox("Choose Your Mission", ["Fintech Galaxy", "Time Traveler's Portfolio"])
st.markdown('</div>', unsafe_allow_html=True)

# Fintech Galaxy Page
if page == "Fintech Galaxy":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("MISSION: MAP THE FINTECH GALAXY")
    
    # Level 1: Load Data
    st.subheader("Level 1: Link Star Map")
    if st.session_state.current_level["galaxy"] >= 1:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Load your data!</span></div> Upload or fetch.', unsafe_allow_html=True)
        if st.button("Load Data"):
            try:
                if data_source == "Upload Kragle Dataset" and galaxy_uploaded_file is not None:
                    st.session_state.galaxy_data = pd.read_csv(galaxy_uploaded_file)
                    if st.session_state.galaxy_data.empty:
                        raise ValueError("Uploaded CSV is empty!")
                    st.session_state.fincoins += 10
                    st.session_state.current_level["galaxy"] = 2
                    st.success("✅ Star Map Linked! +10 FinCoins")
                elif data_source == "Fetch Yahoo Finance Data" and galaxy_ticker:
                    stock = yf.Ticker(galaxy_ticker)
                    st.session_state.galaxy_data = stock.history(start=galaxy_start_date, end=galaxy_end_date)
                    if st.session_state.galaxy_data.empty:
                        raise ValueError(f"No data found for {galaxy_ticker} in the given date range!")
                    st.session_state.fincoins += 10
                    st.session_state.current_level["galaxy"] = 2
                    st.success(f"✅ {galaxy_ticker} Star Map Loaded! +10 FinCoins")
                else:
                    st.error("Drop a file or pick a planet!")
                
                if st.session_state.galaxy_data is not None and not st.session_state.galaxy_data.empty:
                    st.write("Star Map Preview:")
                    st.dataframe(st.session_state.galaxy_data.head())
                    missing_vals = st.session_state.galaxy_data.isnull().sum()
                    fig = px.bar(x=missing_vals.index, y=missing_vals.values, title="Missing Values by Column",
                                 color_discrete_sequence=["#FF007A"], labels={"x": "Columns", "y": "Missing Values"})
                    fig.update_layout(template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"💥 Data Load Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 2: Preprocessing
    st.subheader("Level 2: Clear Cosmic Junk")
    if st.session_state.current_level["galaxy"] >= 2:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Clean data!</span></div> Wipe out black holes.', unsafe_allow_html=True)
        if st.button("Preprocess Data") and st.session_state.galaxy_data is not None:
            try:
                df = st.session_state.galaxy_data.copy()
                missing_values = df.isnull().sum()
                st.write("Black Holes (Missing Values):")
                st.write(missing_values)
                
                df = df.fillna(method="ffill").fillna(method="bfill")
                if "Close" in df.columns:
                    q1 = df["Close"].quantile(0.25)
                    q3 = df["Close"].quantile(0.75)
                    iqr = q3 - q1
                    df = df[(df["Close"] >= q1 - 1.5 * iqr) & (df["Close"] <= q3 + 1.5 * iqr)]
                else:
                    raise KeyError("Close column not found in data!")
                
                st.session_state.galaxy_data = df
                st.session_state.fincoins += 15
                st.session_state.current_level["galaxy"] = 3
                st.success("🧹 Junk Cleared! +15 FinCoins")
                st.write("Cleaned Star Map Preview:")
                st.dataframe(df.head())
                fig = px.box(df, y="Close", title="Cleaned Data Distribution", color_discrete_sequence=["#00FFE5"])
                fig.update_layout(template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"💥 Preprocessing Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 3: Feature Engineering
    st.subheader("Level 3: Mine Dope Features")
    if st.session_state.current_level["galaxy"] >= 3:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Extract metrics!</span></div> Dig for stats.', unsafe_allow_html=True)
        if st.button("Mine Features") and st.session_state.galaxy_data is not None:
            try:
                df = st.session_state.galaxy_data.copy()
                if "Close" in df.columns:
                    df["Returns"] = df["Close"].pct_change()
                    df["Volatility"] = df["Close"].rolling(window=20).std()
                    df["MA_20"] = df["Close"].rolling(window=20).mean()
                    
                    features = ["Returns", "Volatility", "MA_20"]
                    st.session_state.galaxy_features = features
                    df = df.dropna()
                    st.session_state.galaxy_data = df
                    
                    st.session_state.fincoins += 15
                    st.session_state.current_level["galaxy"] = 4
                    st.success("⛏️ Feature Hustler Unlocked! +15 FinCoins")
                    st.write("Dope Features:", features)
                    
                    corr = df[features].corr()
                    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index, y=corr.columns, colorscale='Viridis',
                                                   text=corr.round(2).values, texttemplate="%{text}", colorbar_title="Correlation"))
                    fig.update_layout(title="Stellar Feature Heatmap", template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No Close price in star map!")
            except Exception as e:
                st.error(f"💥 Feature Engineering Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 4: Train/Test Split
    st.subheader("Level 4: Split the Crew")
    if st.session_state.current_level["galaxy"] >= 4:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Prepare clustering!</span></div> Divide forces.', unsafe_allow_html=True)
        if st.button("Split Data") and st.session_state.galaxy_data is not None and st.session_state.galaxy_features:
            try:
                df = st.session_state.galaxy_data
                X = df[st.session_state.galaxy_features]
                scaler = StandardScaler()
                st.session_state.X_scaled = scaler.fit_transform(X)
                
                split_data = {"Training Data": 80, "Validation Data": 20}
                fig = px.pie(values=list(split_data.values()), names=list(split_data.keys()), 
                             title="Crew Split", color_discrete_sequence=["#FF007A", "#00FFE5"])
                fig.update_layout(template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                st.plotly_chart(fig, use_container_width=True)
                
                fig_scatter = px.scatter(df, x="Returns", y="Volatility", size="MA_20", color_discrete_sequence=["#FFFF00"],
                                        title="Scaled Feature Scatter", hover_data=["MA_20"])
                fig.update_layout(template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.session_state.fincoins += 10
                st.session_state.current_level["galaxy"] = 5
                st.success("🌌 Crew Split! +10 FinCoins")
            except Exception as e:
                st.error(f"💥 Data Split Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 5: Model Training with Cluster Guess
    st.subheader("Level 5: Spot the Factions")
    if st.session_state.current_level["galaxy"] >= 5:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Guess factions!</span></div> How many crews?', unsafe_allow_html=True)
        st.session_state.cluster_guess = st.number_input("Guess Factions (Clusters)", min_value=2, max_value=10, value=3)
        if st.button("Train K-Means") and st.session_state.X_scaled is not None:
            try:
                model = KMeans(n_clusters=int(st.session_state.cluster_guess), random_state=42)
                st.session_state.labels = model.fit_predict(st.session_state.X_scaled)
                st.session_state.galaxy_model = model
                
                cluster_sizes = np.bincount(st.session_state.labels)
                fig = px.bar(x=range(len(cluster_sizes)), y=cluster_sizes, title="Cluster Sizes",
                             color_discrete_sequence=["#FF007A"], labels={"x": "Cluster", "y": "Size"})
                fig.update_layout(template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.fincoins += 15
                st.session_state.current_level["galaxy"] = 6
                st.success("🪐 Factions Spotted! +15 FinCoins")
            except Exception as e:
                st.error(f"💥 K-Means Training Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 6: Evaluation
    st.subheader("Level 6: Check Faction Vibes")
    if st.session_state.current_level["galaxy"] >= 6:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Evaluate clusters!</span></div> How tight?', unsafe_allow_html=True)
        if st.button("Evaluate Vibes") and st.session_state.galaxy_model is not None:
            try:
                silhouette = silhouette_score(st.session_state.X_scaled, st.session_state.labels)
                st.metric("Vibe Score (Silhouette)", f"{silhouette:.2f}", delta_color="normal")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                y_lower = 10
                for i in range(st.session_state.galaxy_model.n_clusters):
                    ith_cluster_silhouette_values = silhouette_samples(st.session_state.X_scaled, st.session_state.labels)[st.session_state.labels == i]
                    ith_cluster_silhouette_values.sort()
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    color = plt.cm.nipy_spectral(float(i) / st.session_state.galaxy_model.n_clusters)
                    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
                    y_lower = y_upper + 10
                ax.set_title("Silhouette Plot", color="#00FFE5")
                ax.set_xlabel("Silhouette Coefficient")
                ax.set_ylabel("Cluster Label")
                ax.axvline(x=silhouette, color="red", linestyle="--")
                ax.set_facecolor('#1C2526')
                fig.patch.set_facecolor('#1C2526')
                ax.tick_params(colors='#FFFFFF')
                st.pyplot(fig)
                
                if silhouette >= 0.6:
                    st.session_state.fincoins += 20
                    st.success("🌟 Dope Vibes! Commander Status Unlocked! +20 FinCoins")
                    st.image("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbHlnNDZvNmMxcDZhMnRhdDZsZWRzb2E0N3ZvN21jNGh0aDcybDdvciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o6ZtjxN1PH0urZCso/giphy.gif", caption="[ICON: Commander Badge]", width=100)
                
                actual_clusters = len(set(st.session_state.labels))
                if int(st.session_state.cluster_guess) == actual_clusters:
                    st.session_state.fincoins += 5
                    st.success("🎉 Nailed the Guess! +5 FinCoins")
                
                st.session_state.current_level["galaxy"] = 7
            except Exception as e:
                st.error(f"💥 Evaluation Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 7: Results Visualization
    st.subheader("Level 7: Map the Galaxy")
    if st.session_state.current_level["galaxy"] >= 7:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Visualize galaxy!</span></div> Drop the starfield.', unsafe_allow_html=True)
        if st.button("Drop the Map") and st.session_state.labels is not None:
            try:
                df = st.session_state.galaxy_data.copy()
                df["Cluster"] = st.session_state.labels
                
                fig_2d = px.scatter(df, x="Returns", y="Volatility", color="Cluster", 
                                    title="Dope Galaxy Starfield (2D)", size="MA_20", 
                                    color_continuous_scale=["#FF007A", "#00FFE5", "#FFFF00"], 
                                    hover_data=["MA_20"])
                fig_2d.update_layout(template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                st.plotly_chart(fig_2d, use_container_width=True)
                
                fig_3d = px.scatter_3d(df, x="Returns", y="Volatility", z="MA_20", color="Cluster",
                                       title="Dope Galaxy Starfield (3D)", size_max=10,
                                       color_continuous_scale=["#FF007A", "#00FFE5", "#FFFF00"])
                fig_3d.update_layout(template="plotly_dark", scene=dict(bgcolor="#1C2526"), paper_bgcolor="#1C2526", font_color="#FFFFFF")
                st.plotly_chart(fig_3d, use_container_width=True)
                
                cluster_counts = df["Cluster"].value_counts()
                fig_pie = px.pie(names=cluster_counts.index, values=cluster_counts.values, title="Cluster Distribution",
                                 color_discrete_sequence=["#FF007A", "#00FFE5", "#FFFF00"])
                fig_pie.update_layout(template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                st.plotly_chart(fig_pie, use_container_width=True)
                
                results_df = df[st.session_state.galaxy_features + ["Cluster"]]
                csv = results_df.to_csv(index=False)
                st.download_button("Grab Star Map", csv, "fintech_galaxy_clusters.csv", "text/csv")
                
                if st.session_state.fincoins >= 100:
                    rank = "🌌 Time Commander"
                elif st.session_state.fincoins >= 60:
                    rank = "🚀 Time Explorer"
                else:
                    rank = "🚧 Cadet Analyst"
                st.metric("Your Rank", rank, delta_color="normal")
                
                st.session_state.fincoins += 10
                st.success("🌠 Galaxy Mapped! +10 FinCoins")
                st.image("https://media.giphy.com/media/l0MYt5jPRbrfiO4s8/giphy.gif", caption="[ICON: Mission Complete]", width=400)
                
                if silhouette < 0.6:
                    if st.button("Remix the Map?"):
                        st.session_state.current_level["galaxy"] = 5
                        st.session_state.fincoins += 5
                        st.info("🔁 Remix Unlocked! +5 FinCoins")
            except Exception as e:
                st.error(f"💥 Visualization Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    st.markdown('</div>', unsafe_allow_html=True)

# Time Traveler's Portfolio Page
elif page == "Time Traveler's Portfolio":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("MISSION: PREDICT THE FUTURE")
    
    # Level 1: Load Data
    st.subheader("Level 1: Boot Time Machine")
    if st.session_state.current_level["portfolio"] >= 1:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Load stock data!</span></div> Input or upload.', unsafe_allow_html=True)
        if simulate or portfolio_uploaded_file is not None:
            try:
                if portfolio_uploaded_file:
                    data = pd.read_csv(portfolio_uploaded_file)
                    if 'Date' not in data.columns or 'Close' not in data.columns:
                        raise ValueError("CSV must contain 'Date' and 'Close' columns!")
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.sort_values('Date', inplace=True)
                    st.session_state.portfolio_data = data
                    st.session_state.fincoins += 10
                    st.session_state.current_level["portfolio"] = 2
                    st.success("✅ Time Machine Booted! +10 FinCoins")
                else:
                    data = yf.download(portfolio_ticker, start=portfolio_start_date)
                    if data.empty:
                        raise ValueError(f"No data for {portfolio_ticker} from {portfolio_start_date}!")
                    data = data[['Open', 'High', 'Low', 'Close']].dropna()
                    data['Date'] = data.index
                    data.reset_index(drop=True, inplace=True)
                    st.session_state.portfolio_data = data
                    st.session_state.fincoins += 10
                    st.session_state.current_level["portfolio"] = 2
                    st.success(f"✅ {portfolio_ticker} Data Loaded! +10 FinCoins")
                
                if st.session_state.portfolio_data is not None and not st.session_state.portfolio_data.empty:
                    st.write("Time Data Preview:")
                    st.dataframe(st.session_state.portfolio_data.head())
            except Exception as e:
                st.error(f"💥 Time Data Load Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 2: Preprocessing
    st.subheader("Level 2: Calibrate Time Circuits")
    if st.session_state.current_level["portfolio"] >= 2:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Clean data!</span></div> Fix time anomalies.', unsafe_allow_html=True)
        if st.button("Calibrate Data") and st.session_state.portfolio_data is not None:
            try:
                df = st.session_state.portfolio_data.copy()
                missing_values = df.isnull().sum()
                st.write("Time Anomalies (Missing Values):")
                st.write(missing_values)
                
                df = df.fillna(method="ffill").fillna(method="bfill")
                st.session_state.portfolio_data = df
                st.session_state.fincoins += 15
                st.session_state.current_level["portfolio"] = 3
                st.success("🛠️ Circuits Calibrated! +15 FinCoins")
                st.write("Calibrated Data Preview:")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"💥 Preprocessing Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 3: Model Training
    st.subheader("Level 3: Train Time Predictor")
    if st.session_state.current_level["portfolio"] >= 3:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Train model!</span></div> Predict future prices.', unsafe_allow_html=True)
        if st.button("Train Predictor") and st.session_state.portfolio_data is not None:
            try:
                df = st.session_state.portfolio_data
                df['Day'] = np.arange(len(df))
                X = df[['Day']]
                y = df['Close']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression().fit(X_train, y_train)
                st.session_state.portfolio_model = model
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
                st.session_state.y_train, st.session_state.y_test = y_train, y_test
                
                st.session_state.fincoins += 15
                st.session_state.current_level["portfolio"] = 4
                st.success("🔮 Predictor Trained! +15 FinCoins")
            except Exception as e:
                st.error(f"💥 Model Training Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 4: Evaluation and Portfolio Simulation
    st.subheader("Level 4: Simulate Portfolio")
    if st.session_state.current_level["portfolio"] >= 4:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Simulate investment!</span></div> Check gains.', unsafe_allow_html=True)
        if st.button("Simulate Portfolio") and st.session_state.portfolio_data is not None:
            try:
                df = st.session_state.portfolio_data
                initial_price = float(df.iloc[0]['Close'])
                final_price = float(df.iloc[-1]['Close'])
                shares = investment_amount / initial_price
                current_value = shares * final_price
                roi = ((current_value - investment_amount) / investment_amount) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📉 Initial Price", f"${initial_price:.2f}")
                col2.metric("📈 Final Price", f"${final_price:.2f}")
                col3.metric("📊 ROI", f"{roi:.2f}%", delta=f"{final_price - initial_price:.2f}")
                
                st.info(f"💰 You own **{shares:.2f} shares** of **{portfolio_ticker.upper()}**")
                st.success(f"💸 Portfolio Value: **${current_value:.2f}**")
                
                mse = mean_squared_error(st.session_state.y_test, st.session_state.portfolio_model.predict(st.session_state.X_test))
                st.metric("Model MSE", f"{mse:.2f}", delta_color="normal")
                
                if mse < df['Close'].std() ** 2:
                    st.session_state.fincoins += 20
                    st.success("🌟 Tight Prediction! Time Explorer Badge Unlocked! +20 FinCoins")
                    st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZjlvdmF0MzY4ZWtiN2U3YnQxeHBhYWowMmc5eWRqY2p4cnh3bXVxayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bztUNMLxXzSaQ0SvHv/giphy.gif", caption="[ICON: Time Explorer]", width=100)
                
                st.session_state.fincoins += 15
                st.session_state.current_level["portfolio"] = 5
                st.success("📈 Portfolio Simulated! +15 FinCoins")
            except Exception as e:
                st.error(f"💥 Simulation Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    
    # Level 5: Results Visualization and Prediction
    st.subheader("Level 5: See the Future")
    if st.session_state.current_level["portfolio"] >= 5:
        st.markdown('<div class="tooltip"><span class="icon-placeholder"></span><span class="tooltiptext">Visualize future!</span></div> Drop the forecast.', unsafe_allow_html=True)
        if st.button("Drop Forecast"):
            try:
                df = st.session_state.portfolio_data
                initial_price = float(df.iloc[0]['Close'])
                
                # Growth Chart
                df['Growth %'] = ((df['Close'] - initial_price) / initial_price) * 100
                fig_growth = go.Figure()
                fig_growth.add_trace(go.Scatter(x=df['Date'], y=df['Growth %'], mode='lines', name='Growth %', line=dict(color="#FF007A")))
                fig_growth.update_layout(title="Investment Growth (%)", xaxis_title="Date", yaxis_title="Growth %",
                                         template="plotly_dark", plot_bgcolor="#1C2526", paper_bgcolor="#1C2526", font_color="#FFFFFF")
                st.plotly_chart(fig_growth, use_container_width=True)
                
                # Future Prediction
                future_day_index = len(df) + future_days
                future_price = float(st.session_state.portfolio_model.predict(np.array([[future_day_index]]))[0])
                st.success(f"📅 In {future_days} days, {portfolio_ticker.upper()} predicted: **${future_price:.2f}**")
                
                # Download Forecast
                last_date = df['Date'].dropna().iloc[-1] if not df['Date'].dropna().empty else df.index[-1]
                future_days_range = np.arange(len(df), len(df)+180).reshape(-1, 1)
                future_prices = st.session_state.portfolio_model.predict(future_days_range).flatten()
                forecast_df = pd.DataFrame({
                    'Date': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180),
                    'Predicted Price': future_prices
                })
                csv_forecast = forecast_df.to_csv(index=False)
                st.download_button("Grab 180-day Forecast", csv_forecast, "future_forecast.csv", "text/csv")
                
                # Correlation Heatmap
                if not portfolio_uploaded_file:
                    st.subheader("📈 Correlation with Dope Stocks")
                    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                    stock_data = yf.download(tech_stocks, start=portfolio_start_date)['Close']
                    stock_data = stock_data.dropna()
                    correlation = stock_data.pct_change().corr()
                    fig_corr, ax = plt.subplots(figsize=(8, 6), facecolor='#1C2526')
                    sns.heatmap(
                        correlation, annot=True, cmap='mako', linewidths=0.5, linecolor='#1C2526',
                        cbar_kws={"shrink": 0.7, 'label': 'Correlation'}, annot_kws={"color": "white", "size": 10}, ax=ax
                    )
                    ax.set_facecolor('#1C2526')
                    fig_corr.patch.set_facecolor('#1C2526')
                    ax.tick_params(colors='#FFFFFF')
                    ax.set_title("Tech Stock Correlation Matrix", color='#00FFE5', fontsize=14)
                    st.pyplot(fig_corr)
                
                # Investment Advice
                advice = ["Buy the dip! 🕳️📉", "Time in > timing! ⏳📈", "Zoom out, chill! 📆",
                          "Buffet sleeps, you should too! 🛌", "Diversify, fam! 🤔"]
                st.info(np.random.choice(advice))
                
                st.session_state.fincoins += 10
                st.success("🌌 Future Dropped! +10 FinCoins")
                st.image("https://media.giphy.com/media/l0MYt5jPRbrfiO4s8/giphy.gif", caption="[ICON: Time Mission Complete]", width=400)
                
                if mse >= df['Close'].std() ** 2:
                    if st.button("Retry Prediction?"):
                        st.session_state.current_level["portfolio"] = 3
                        st.session_state.fincoins += 5
                        st.info("🔁 Retry Unlocked! +5 FinCoins")
            except Exception as e:
                st.error(f"💥 Visualization Error: {e}")
    else:
        st.warning("Unlock previous levels!")
    st.markdown('</div>', unsafe_allow_html=True)

# Fun Fact Redemption
st.sidebar.markdown('<div class="section">', unsafe_allow_html=True)
st.sidebar.subheader("Cash In FinCoins")
if st.sidebar.button("Score Fun Fact (10 FinCoins)"):
    if st.session_state.fincoins >= 10:
        st.session_state.fincoins -= 10
        st.sidebar.write("🌟 Fun Fact: Fintech investments hit $210B in 2023!")
        st.sidebar.image("https://media.giphy.com/media/3o7TKsQ8kB3Y4X2Z8I/giphy.gif", caption="[ICON: Fun Fact]", width=100)
    else:
        st.sidebar.error("Need more FinCoins!")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

