import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import black_litterman_model as blm

#Styling of page
st.set_page_config(
    page_title="SIMM Risk Engine", 
    layout="wide",
    page_icon="📈"
)


st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp {
        background-color: #0e1117; /* Dark Slate */
        color: #e0e0e0;
    }
    
    /* 2. Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* 3. Containers/Cards (Simulating a 'Card' look) */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    
    /* 4. Typography */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stCaption {
        color: #8b949e !important;
        font-size: 14px;
    }
    
    /* 5. Inputs & Selectboxes */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stTextInput input, 
    .stNumberInput input {
        background-color: #0d1117 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 6px;
    }
    
    /* 6. Buttons */
    .stButton > button {
        background-color: #238636 !important; /* GitHub/Finance Green */
        color: white !important;
        border: 1px solid rgba(240, 246, 252, 0.1) !important;
        border-radius: 6px;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #2ea043 !important;
        border-color: #8b949e !important;
    }
    
    /* 7. Tables */
    [data-testid="stDataFrame"] {
        background-color: #0d1117;
    }
    </style>
    """, unsafe_allow_html=True)

#Pulling functions from black_litterman_model
RISK_FREE_RATE = 0.0421 # 4.21%
SECTOR_MAP = {
    'XLK': 'Technology', 'XLP': 'Cons. Staples', 'XLB': 'Materials',
    'XLF': 'Financials', 'XLV': 'Healthcare', 'XLU': 'Utilities',
    'XLI': 'Industrials', 'AGG': 'Fixed Income'
}
#Sets the time window to gather price history, and it is dynamic based on the day you run this program
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = blm.sectors
    end_date = datetime.date.today()
    #Dynamic 20-Year Window
    start_date = end_date - datetime.timedelta(days=365*20)
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        try: price_table = data['Close']
        except KeyError: price_table = data.xs('Close', level=1, axis=1)
    else:
        price_table = data['Close']
    return price_table.reindex(columns=sorted(tickers))

#Creating the sidebar where you can adjust market baseline
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Set your market baseline parameters.")
    
    baseline_option = st.selectbox(
        "Benchmark Strategy",
        ("SIMM Benchmark", "90/10", "80/20", "60/40")
    )
    
    st.divider()
    
    #Live Metrics in Sidebar
    st.metric("Risk Free Rate", f"{RISK_FREE_RATE:.2%}")
    st.caption("Using 10Y Treasury Yield Assumption")

#Using price data and beginning calculations of mu_bl
with st.spinner('📡 Connecting to Market Data...'):
    prices = get_market_data()
    log_returns = blm.log_returns(prices)
    cov_matrix = blm.cov_matrix(log_returns)

    #Determine Weights based on Sidebar
    if baseline_option == "90/10": weights = blm.ninety_ten(cov_matrix)
    elif baseline_option == "80/20": weights = blm.eighty_twenty(cov_matrix)
    elif baseline_option == "60/40": weights = blm.sixty_fourty(cov_matrix)
    else: weights = blm.simm_bench(cov_matrix)

    #Calculate Market Baseline
    var = blm.benchmark_variance(weights, cov_matrix)
    lam = blm.lambda_risk_aversion(var, str(baseline_option))
    pi = blm.implied_returns(lam, cov_matrix, weights)

#Main content page of webpage

# Title Section
c1, c2 = st.columns([1, 5])
with c1:
    #Optional: Display Logo if file exists, else icon
    try: st.image("RiskLogo.png")
    except: st.header("📈")
with c2:
    st.title("SIMM Black-Litterman Engine")
    st.caption(f"Optimization Model • {baseline_option} Baseline")

st.markdown("---")

#Layout: Two Columns (Market Data vs. Inputs)
left_col, right_col = st.columns([1, 1])

#Market context
with left_col:
    st.subheader("1. Market Implied Returns")
    st.caption("Returns the market expects based on current prices (Prior).")
    
    implied_df = pd.DataFrame({
        "Ticker": pi.index,
        "Sector": [SECTOR_MAP.get(t, t) for t in pi.index],
        "Implied Return": pi.values
    })
    
    #Clean Table
    st.dataframe(
        implied_df.style.format({"Implied Return": "{:.2%}"})
        .background_gradient(subset=["Implied Return"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
        height=300
    )

#Analysts inputs
with right_col:
    st.subheader("2. Analyst Views")
    st.caption("Inject your alpha. What do you know that the market doesn't?")

    if 'views' not in st.session_state: st.session_state.views = []

    #Input Form Container
    with st.container():
        c_in1, c_in2 = st.columns(2)
        with c_in1:
            target = st.selectbox("Win Asset", cov_matrix.index)
        with c_in2:
            relative = st.selectbox("Lose Asset (Optional)", ["None"] + list(cov_matrix.index))
        
        ret_val = st.number_input("Expected Return Magnitude (e.g., 0.05)", value=0.05, step=0.005, format="%.4f")
        
        if st.button("➕ Add View Strategy", use_container_width=True):
            sub = None if relative == "None" else relative
            st.session_state.views.append({'target': target, 'subsidiary': sub, 'return': ret_val})
            st.toast(f"View Added: {target}", icon="✅")

    #View List
    if st.session_state.views:
        st.write("Active Views:")
        view_display = []
        for i, v in enumerate(st.session_state.views):
            desc = f"**{v['target']}** > **{v['subsidiary']}**" if v['subsidiary'] else f"**{v['target']}** Absolute"
            view_display.append({"Strategy": desc, "Return": f"{v['return']:.2%}"})
        
        st.dataframe(pd.DataFrame(view_display), use_container_width=True, hide_index=True)
        
        if st.button("Clear Views", type="secondary"):
            st.session_state.views = []
            st.rerun()

st.markdown("---")

#Displaying results
st.header("3. Optimization Results")

if st.button("🚀 Run Optimization Model", type="primary", use_container_width=True):
    if not st.session_state.views:
        st.warning("⚠️ Please add at least one analyst view above to run the model.")
    else:
        #Calculations
        formatted_views = [{'target': v['target'], 'subsidiary': v['subsidiary'], 'return': v['return']} for v in st.session_state.views]
        sectors_list = list(cov_matrix.columns)
        
        P, Q = blm.view_vectors(formatted_views, sectors_list)
        omega = blm.omega(P, cov_matrix)
        mu_bl = blm.mu_bl(cov_matrix, P, Q, omega, pi)
        
        #Creating a datafram for results
        results = pd.DataFrame(index=cov_matrix.index)
        results['Ticker'] = results.index
        results['Sector'] = [SECTOR_MAP.get(t, t) for t in results.index]
        results['Market Implied'] = pi
        
        #Map Views
        results['Analyst View'] = np.nan
        for v in formatted_views:
            results.loc[v['target'], 'Analyst View'] = v['return']
            
        results['BL Combined'] = mu_bl
        results['Total Exp. Return'] = results['BL Combined'] + RISK_FREE_RATE
        results['Spread'] = results['BL Combined'] - results['Market Implied']
        
        def get_rec(spread):
            if spread > 0.0015: return 'Increase'
            elif spread < -0.0015: return 'Decrease'
            else: return 'Stable'
        results['Recommendation'] = results['Spread'].apply(get_rec)

        #Final display
        final_df = results[[
            'Ticker', 'Sector', 'Market Implied', 'Analyst View', 
            'BL Combined', 'Total Exp. Return', 'Recommendation'
        ]]

        #Styling
        def color_rec(val):
            if val == 'Increase': return 'color: #2ea043; font-weight: bold; background-color: rgba(46, 160, 67, 0.1)'
            if val == 'Decrease': return 'color: #da3633; font-weight: bold; background-color: rgba(218, 54, 51, 0.1)'
            return 'color: #8b949e'

        st.subheader("Allocation Strategy")
        st.caption("Final recommendations based on blended Market Priors and Analyst Views.")

        st.dataframe(
            final_df.style
            .format({
                'Market Implied': '{:.2%}',
                'Analyst View': '{:.2%}',
                'BL Combined': '{:.2%}',
                'Total Exp. Return': '{:.2%}'
            }, na_rep="-")
            .map(color_rec, subset=['Recommendation'])
            .bar(subset=['Total Exp. Return'], color='#1f6feb', vmin=0) # Blue bars for visual magnitude
            , use_container_width=True
            , height=500
        )   
        
# --- 4. BENCHMARK REFERENCE TABLE ---
st.markdown("---")
st.subheader("4. Benchmark Reference Guide")
st.caption("Comparison of target weights across all available strategies.")

# 1. Define the Weights Data (Hardcoded from your module for display)
reference_data = {
    "Ticker": ['XLK', 'XLP', 'XLB', 'XLF', 'XLV', 'XLU', 'XLI', 'AGG', 'AOR'],
    "Sector": ['Technology', 'Cons. Staples', 'Materials', 'Financials', 'Healthcare', 'Utilities', 'Industrials', 'Fixed Income', 'Mixed'],
    "90/10 (Aggressive)": [0.20, 0.10, 0.10, 0.10, 0.10, 0.05, 0.10, 0.10, 0.15],
    "80/20 (Growth)":     [0.18, 0.09, 0.09, 0.09, 0.09, 0.04, 0.09, 0.20, 0.13],
    "60/40 (Balanced)":   [0.14, 0.06, 0.06, 0.07, 0.07, 0.03, 0.07, 0.40, 0.10],
    "SIMM Benchmark":     [0.18, 0.09, 0.09, 0.09, 0.09, 0.04, 0.09, 0.20, 0.13]
}

# 2. Create DataFrame
ref_df = pd.DataFrame(reference_data)

# 3. Styling Logic
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: rgba(46, 160, 67, 0.2)' if v else '' for v in is_max]

st.dataframe(
    ref_df.style
    .format({
        "90/10 (Aggressive)": "{:.1%}",
        "80/20 (Growth)": "{:.1%}",
        "60/40 (Balanced)": "{:.1%}",
        "SIMM Benchmark": "{:.1%}"
    })
    .apply(highlight_max, subset=["90/10 (Aggressive)", "80/20 (Growth)", "60/40 (Balanced)"], axis=1) # Highlights the strategy with the highest weight for that sector
    .background_gradient(subset=["90/10 (Aggressive)", "80/20 (Growth)", "60/40 (Balanced)"], cmap="Greens", vmin=0, vmax=0.4),
    use_container_width=True,
    hide_index=True,
    height=400
)