import streamlit as st
import pandas as pd
import torch

#looks for bo_pipeline.py & imports classes/functions
from BO_pipeline import BODataModule, run_bo_continuous 

# --- 1. CATALYST DATABASE LOADING : to get the featurized name of the catalyst---
@st.cache_data
def load_catalyst_database(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {"🔍 Database not found. Check CSV path...": None}
    
    catalyst_features = [
        'Ni Loading', 'Pore Size', 'Pore Volume', 
        'Surface Area', 'H2-TPR Peak Temperature', 'Ni Particle Size'
    ]
    
    cat_columns = [col for col in df.columns if col.startswith('Cat_')]
    cat_db = {"Custom / Unknown (Let BO optimize)": None}
    
    for cat_col in cat_columns:
        subset = df[df[cat_col] == 1]
        if not subset.empty:
            props = subset[catalyst_features].mean().to_dict()
            clean_name = cat_col.replace("Cat_", "").strip()
            cat_db[clean_name] = props
            
    return cat_db

# Initialize the database
CATALYST_DB = load_catalyst_database("v2c_featurized_ml_ready.csv")

# Initialize session state for default values so the UI updates dynamically
if "default_props" not in st.session_state:
    st.session_state.default_props = {
        'Ni Loading': 10.0, 'Pore Size': 12.0, 'Pore Volume': 0.5, 
        'Surface Area': 150.0, 'H2-TPR Peak Temperature': 600.0, 'Ni Particle Size': 15.0
    }

# --- 2. SETUP PAGE ---
st.set_page_config(page_title="DRM Reactor BO Optimizer", layout="wide")

st.title("🧪 DRM Reactor Bayesian Optimizer")
st.markdown("Set your known constraints and let the AI find the optimal parameters for the rest.")

col1, col2 = st.columns([1, 1.5])

# --- 3. PARAMETER CONFIGURATION (LEFT COLUMN) ---
with col1:
    st.header("⚙️ Configuration")
    fixed_params = {} 

    st.subheader("1. Catalyst System")
    
    # Callback to update default values when catalyst changes
    def on_catalyst_change():
        selected = st.session_state.cat_selector
        if CATALYST_DB[selected] is not None:
            st.session_state.default_props = CATALYST_DB[selected]

    selected_cat = st.selectbox(
        "Choose Catalyst (Auto-fills physical properties):", 
        list(CATALYST_DB.keys()),
        key="cat_selector",
        on_change=on_catalyst_change
    )

    st.write("Lock specific properties, or leave them unlocked for AI optimization.")
    
    # Physical Properties (Smart Defaults + Toggle)
    props = st.session_state.default_props
    
    if st.toggle("Ni Loading", value=True):
        fixed_params['Ni Loading'] = st.number_input("Ni Loading (wt%)", value=float(props['Ni Loading']))
    if st.toggle("Pore Size", value=True):
        fixed_params['Pore Size'] = st.number_input("Pore Size (nm)", value=float(props['Pore Size']))
    if st.toggle("Pore Volume", value=True):
        fixed_params['Pore Volume'] = st.number_input("Pore Volume (cm³/g)", value=float(props['Pore Volume']))
    if st.toggle("Surface Area", value=True):
        fixed_params['Surface Area'] = st.number_input("Surface Area (m²/g)", value=float(props['Surface Area']))
    if st.toggle("H2-TPR Peak Temp", value=True):
        fixed_params['H2-TPR Peak Temperature'] = st.number_input("H2-TPR Peak Temp (°C)", value=float(props['H2-TPR Peak Temperature']))
    if st.toggle("Ni Particle Size", value=True):
        fixed_params['Ni Particle Size'] = st.number_input("Ni Particle Size (nm)", value=float(props['Ni Particle Size']))

    st.subheader("2. Operational Conditions")
    if st.toggle("Ratio of CH4 in Feed", value=False):
        fixed_params['Ratio of CH4 in Feed'] = st.slider("CH4/CO2 Ratio", 0.5, 2.0, 1.0)
    if st.toggle("Reaction Temperature", value=False):
        fixed_params['Reaction Temperature'] = st.slider("Temperature (°C)", 200.0, 1000.0, 750.0)
    if st.toggle("Reaction Time", value=False):
        fixed_params['Reaction Time'] = st.number_input("Time (h)", 1.0, 24.0, 5.0)
    if st.toggle("GHSV", value=False):
        fixed_params['GHSV'] = st.number_input("GHSV (mL/g/h)", 1000.0, 100000.0, 24000.0)


# --- 4. EXECUTION & RESULTS (RIGHT COLUMN) ---
with col2:
    st.image("DRM unit.png", caption="Fixed-Bed Reactor", use_container_width=True)
    
    st.write("### Current Fixed Parameters:")
    st.json(fixed_params)
    
    # Run BO Button
    if st.button("🚀 Run Bayesian Optimization", type="primary", use_container_width=True):
        
        # This spinner shows the user that the background code is running
        with st.spinner('Optimizing Pareto front... This may take a few minutes depending on iterations.'):
            try:
                # Initialize Data Module
                dm = BODataModule(data_path="v2c_featurized_ml_ready.csv", n_init=15, target="both")
                
                # Define Bounds
                bounds = torch.tensor([dm.X.min(axis=0), dm.X.max(axis=0)], dtype=torch.float64)
                
                # Exact feature names expected by your model
                feature_names = [
                    'Ratio of CH4 in Feed', 'Reaction Temperature', 'Ni Loading',
                    'Reaction Time', 'Pore Size', 'Pore Volume', 'Surface Area',
                    'H2-TPR Peak Temperature', 'Ni Particle Size', 'GHSV'
                ]
                
                # Execute the BO Loop!
                train_x, train_y, hv_hist, ch4_hist, co2_hist, next_params = run_bo_continuous(
                    dm=dm, 
                    bounds=bounds, 
                    feature_names=feature_names, 
                    fixed_params=fixed_params,  # Passes the UI toggles into the BO logic
                    num_iterations=20 )          # You can lower this to 10 for faster UI testing # <-- THE PARENTHESIS IS NOW SAFELY ON THIS LINE
                
                st.success("✅ Optimization Complete!")
                
                # Get the absolute best values found during the loop
                best_ch4 = ch4_hist[-1]
                best_co2 = co2_hist[-1]
                
                # Display Results
                st.write("---")
                res_col1, res_col2 = st.columns(2)
                res_col1.metric(label="Max Found CH4 Conversion", value=f"{best_ch4:.2f}%")
                res_col2.metric(label="Max Found CO2 Conversion", value=f"{best_co2:.2f}%")
                
                st.subheader("🎯 Next Best Parameters to Test in the Lab:")
                st.dataframe(pd.DataFrame([next_params]), use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")