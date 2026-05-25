import streamlit as st
import pandas as pd
import torch

#looks for bo_pipeline.py & imports classes/functions
from BO_pipeline import BODataModule, run_bo_continuous 
from catalyst_features import get_catalyst_pcs, pcs_to_catalyst

# --- 1. CATALYST DATABASE LOADING : to get the featurized name of the catalyst---
@st.cache_data
def load_catalyst_list(csv_path):
    try:
        df = pd.read_csv(csv_path, sep=';')
        catalysts = ["Custom / Unknown (Let BO optimize)"] + df['Catalyst'].dropna().unique().tolist()
        return catalysts
    except Exception:
        return ["Custom / Unknown (Let BO optimize)"]

# Initialize the database
CATALYSTS_LIST = load_catalyst_list("Interface/data/cat+chembertaPCA.csv")

# Initialize session state for default values so the UI updates dynamically
if "default_props" not in st.session_state:
    st.session_state.default_props = {
        'Ni Loading': 10.0, 'Pore Size': 12.0, 'Pore Volume': 0.5, 
        'Surface Area': 150.0, 'H2-TPR Peak Temperature': 600.0, 'Ni Particle Size': 15.0
    }

if "bo_results" not in st.session_state:
    st.session_state.bo_results = None


# --- 2. SETUP PAGE ---
st.set_page_config(page_title="DRM Reactor BO Optimizer", layout="wide")

st.image("Interface/Images/reforML.jpg", width=600)

# Create tabs
tab_home, tab_config, tab_results = st.tabs([":material/home: Home", ":material/settings: Configuration", ":material/analytics: Results"])


# --- 3. PRESENTATION TAB (HOME) ---
with tab_home:    
    st.header("Welcome to the DRM Reactor BO Optimizer")
    col_text, col_img = st.columns(2)
    with col_text:
        st.markdown("Lock your Dry Reforming of Methane (DRM) process constraints to guide the Bayesian optimization and find the next best-suited parameters to explore.")
        st.markdown("""
    **How it works:**
    * :material/settings: **Configure:** Define known catalyst properties and operational limits.
    * :material/psychology: **Optimize:** Let the ML model explore the parameter space.
    * :material/analytics: **Discover:** Find the exact conditions for >95% CH4/CO2 conversion.
    """)
        st.info("Head over to the **Configuration** tab to set your parameters and run the optimization!")
    with col_img:
        st.image("Interface/Images/DRM unit.png", caption="Fixed-Bed Reactor", use_container_width=True)


# --- 4. PARAMETER CONFIGURATION TAB ---
with tab_config:
    st.header(":material/settings: Configuration of DRM reactor")
    
    col_cat, col_op = st.columns(2)
    fixed_params = {}

    with col_cat:
        st.subheader("1. Catalyst System")

        selected_cat = st.selectbox(
                "Search or select Catalyst (Type to autofill):", 
                CATALYSTS_LIST,
                key="cat_selector",
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

    with col_op:
        st.subheader("2. Operational Conditions")
        if st.toggle("Ratio of CH4 in Feed", value=False):
            fixed_params['Ratio of CH4 in Feed'] = st.slider("CH4/CO2 Ratio", 0.05, 0.5, step= 0.05)
        if st.toggle("Reaction Temperature", value=False):
            fixed_params['Reaction Temperature'] = st.slider("Temperature (°C)", 200.0, 1000.0, step= 50.0)
        if st.toggle("Reaction Time", value=False):
            fixed_params['Reaction Time'] = st.number_input("Time (h)", 0.1, 200.0)
        if st.toggle("GHSV", value=False):
            fixed_params['GHSV'] = st.number_input("GHSV (mL/g/h)", 1700.0, 1440000.0, step = 1000.0)

        # Fetch PCs for the selected catalyst and lock them for BO (Moved here for UI display)
        cat_choice = st.session_state.cat_selector
        if cat_choice != "Custom / Unknown (Let BO optimize)":
            csv_path_pca = "Interface/data/cat+chembertaPCA.csv"
            pcs = get_catalyst_pcs(cat_choice, csv_path_pca)
            if "error" not in pcs:
                # Add PC1 to PC10 to fixed_params
                for pc_key, pc_val in pcs.items():
                    fixed_params[pc_key] = pc_val

        st.write("### Current Fixed Parameters:")
        # Hide PCs from the UI JSON and show the Catalyst name instead
        display_params = {k: v for k, v in fixed_params.items() if not k.startswith('PC')}
        if cat_choice != "Custom / Unknown (Let BO optimize)":
            display_params['Catalyst'] = cat_choice
        st.json(display_params)

    
    # Run BO Button
    if st.button(":material/bolt: Run Bayesian Optimization", type="primary", use_container_width=True):
        
        # This spinner shows the user that the background code is running
        with st.spinner('Optimizing Pareto front... This may take a few minutes depending on iterations.'):
            try:
                # Initialize Data Module with fixed parameters 
                dm = BODataModule(
                    data_path="Interface/data/v2c_featurized_ml_ready.csv", 
                    n_init=15, 
                    target="both",
                    max_feed=fixed_params.get('Ratio of CH4 in Feed'),
                    max_temp=fixed_params.get('Reaction Temperature'),
                    max_loading=fixed_params.get('Ni Loading'),
                    max_time=fixed_params.get('Reaction Time'),
                    max_pore_size=fixed_params.get('Pore Size'),
                    max_pore_volume=fixed_params.get('Pore Volume'),
                    max_surface_area=fixed_params.get('Surface Area'),
                    max_H2_TPR_peak_temp=fixed_params.get('H2-TPR Peak Temperature'),
                    max_particle_size=fixed_params.get('Ni Particle Size'),
                    max_ghsv=fixed_params.get('GHSV')
                )
                
                # Fetch PCs for the selected catalyst and lock them for BO
                cat_choice = st.session_state.cat_selector
                csv_path_pca = "Interface/data/cat+chembertaPCA.csv"
                if cat_choice != "Custom / Unknown (Let BO optimize)":
                    pcs = get_catalyst_pcs(cat_choice, csv_path_pca)
                    if "error" not in pcs:
                        # Add PC1 to PC10 to fixed_params
                        for pc_key, pc_val in pcs.items():
                            fixed_params[pc_key] = pc_val

                # Define Bounds
                bounds = torch.tensor([dm.X.min(axis=0), dm.X.max(axis=0)], dtype=torch.float64)
                
                # Exact feature names expected by model (including PCs)
                feature_names = [
                    'Ratio of CH4 in Feed', 'Reaction Temperature', 'Ni Loading',
                    'Reaction Time', 'Pore Size', 'Pore Volume', 'Surface Area',
                    'H2-TPR Peak Temperature', 'Ni Particle Size', 'GHSV',
                    'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'
                ]
                
                # Execute the BO Loop!
                train_x, train_y, hv_hist, ch4_hist, co2_hist, next_params = run_bo_continuous(
                    dm=dm, 
                    bounds=bounds, 
                    feature_names=feature_names, 
                    fixed_params=fixed_params,  # Passes the UI toggles into the BO logic
                    num_iterations=20 )          # You can lower this to 10 for faster UI testing # <-- THE PARENTHESIS IS NOW SAFELY ON THIS LINE
                
                # Get the absolute best values found during the loop
                best_ch4 = ch4_hist[-1]
                best_co2 = co2_hist[-1]
                
                # Save results in session state
                st.session_state.bo_results = {
                    "best_ch4": best_ch4,
                    "best_co2": best_co2,
                    "next_params": next_params
                }
                
                st.success(":material/verified: Optimization Complete! Go to the 'Results' tab to see the outcome.")


            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")

# --- 5. RESULTS TAB ---
with tab_results:
    st.header("Optimized Results")
    
    if st.session_state.bo_results is not None:
        res = st.session_state.bo_results
        
        st.write("---")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric(label="Max Found CH4 Conversion", value=f"{res['best_ch4']:.2f}%")
        res_col2.metric(label="Max Found CO2 Conversion", value=f"{res['best_co2']:.2f}%")
        
        st.subheader(":material/tune: Next Best Parameters to Test in the Lab")
        
        # Hide PCs from the final parameter recommendation
        display_res = {k: v for k, v in res['next_params'].items() if not k.startswith('PC')}
        st.dataframe(pd.DataFrame([display_res]), use_container_width=True, hide_index=True)
        
        cat_choice = st.session_state.cat_selector
        if cat_choice != "Custom / Unknown (Let BO optimize)":
            st.info(f":material/check_circle: **Catalyst Maintained:** {cat_choice} (Properties locked for optimization)")
        else:
            # Find closest known catalyst to these optimized PCs
            pcs_opt = {k: v for k, v in res['next_params'].items() if k.startswith('PC')}
            if pcs_opt:
                csv_path_pca = "Interface/data/cat+chembertaPCA.csv"
                closest = pcs_to_catalyst(pcs_opt, csv_path_pca)
                if "error" not in closest:
                    st.info(f":material/lightbulb: The closest known catalyst structure to these optimal PCs is: **{closest['catalyst']}** (Distance: {closest['distance']:.2f})")
                    
                    with st.expander("How to interpret this distance?"):
                        st.markdown("""
                        This value represents the Euclidean distance across 10 structural dimensions (Principal Components) between the AI's theoretical optimum and the actual catalyst.
                        
                        * **Distance < 1.0**: **Very close.** The found catalyst has a nearly identical chemical signature to the theoretical optimum.
                        * **Distance 1.0 to 2.0**: **Moderately close.** The catalyst belongs to a similar structural family, but has some notable differences.
                        * **Distance > 2.5**: **Significantly different.** The AI is suggesting a novel structure. The displayed catalyst is just the "closest available" in the database, but is chemically quite distinct.
                        """)
                
    else:
        st.info("No results to display yet. Please run the Bayesian Optimization in the **Configuration** tab.")
