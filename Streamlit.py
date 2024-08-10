import streamlit as st
import pandas as pd
import numpy as np
# from scipy.stats import norm
# import plotly.graph_objects as go
from numpy import log, sqrt, exp  # Make sure to import these
import matplotlib.pyplot as plt
# import seaborn as sns

# TO RUN STREAMLIT CODE FROM TERMINAL streamlit run your_code.py   (ensure the file path is correct)
#   streamlit run Desktop\Python\Projects\Problem_1\Streamlit.py

#######################
# Page configuration
st.set_page_config(
    page_title="Monte Carlo Simulation of the Heston Model for Option Pricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Sidebar for User Inputs
with st.sidebar:
    st.title("Monte Carlo Simulation of the Heston Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/grant-doherty201a/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Grant Doherty`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Initial Asset Price $S_0$ (Dollars)", value=100.0)
    # strike = st.number_input("Strike Price $K$ (Dollars)", value=100.0)
    volatility = st.number_input("Initial Volatility $v_0$ (Years$^{-1}$)", value=0.2)
    time_to_maturity = st.number_input("Time to Maturity $T$ (Years)", value=1.0)
    mean_variance = st.number_input("Long-tem mean of the volatility θ (Years$^{-1}$)")
    interest_rate = st.number_input("Risk-Free Interest Rate $r$ (%/Year)", value=0.05)
    wiener_correlation = st.number_input("Wiener Correlation Factor ρ")

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)



# def Heston_Sim(S0, v0, r, k, theta, sigma, rho, T, N_steps=100, N_sims=200):

#     """
#     Performs a Monte Carlo simulation of the Heston Model under risk-neutral dynamics to price vanilla options. 

#     Args:   
#             S0: Initial asset price (dollars)
#             v0: Initial asset volatility ()
#             r: Risk-free rate
#             k (kappa): Velocity of mean reversion of the variance
#             theta: Long-term variance mean
#             sigma: Volatility of the variance (vol of vol)
#             rho: Correlation factor between the asset price and variance wiener processes WS & Wv

#     Output: 
#             v_T: Average asset variance(type: Numpy array, Shape: (N_simulations, N_steps))
#             S_T: Average asset price at maturity (type: Numpy array, Shape: (N_simulations, N_steps))
#             timeline: Simulation time array (type: Numpy array, size(N_steps), bounds(0, T))
#     """

#     # Initialize variance and asset price arrays, and set initial conditions
#     v = np.zeros((N_sims, N_steps))
#     S = np.zeros((N_sims, N_steps))
#     v[:, 0], S[:,0] = v0, S0

#     timeline = np.linspace(0, T, N_steps)
#     dt = T/(N_steps-1)

#     mu = np.array([0, 0])
#     cov = np.matrix([[1, rho],
#                     [rho, 1]])
#     W = ss.multivariate_normal.rvs(mean=mu, cov=cov, size=(N_sims, N_steps - 1))    # N_steps-1 since v0 and S0 are accounted for. 
#     WS = W[:, :, 0]  # Stock Brownian motion:     W_1
#     Wv = W[:, :, 1]  # Variance Brownian motion:  W_2

#     for j in range(N_sims):
#         for idx, i in enumerate(timeline[1:]):
#             v[j, idx+1] = np.maximum(v[j,idx] + k*(theta-v[j,idx])*dt + sigma*np.sqrt(v[j,idx]*dt) * Wv[j,idx], 0)
#             S[j, idx+1] = S[j,idx] * np.exp( (r - 0.5*v[j,idx])*dt + np.sqrt(v[j,idx]*dt)*WS[j,idx] )

# #     print(f"Final Variance: {np.mean(v[:,-1])} per unit time")
# #     print(f"Final Mean Asset Price: ${np.mean(S[:,-1])}")

#     S_T, v_T = np.mean(S[:,-1]), np.mean(v[:,-1])

#     return S_T, v, timeline




# def heat_maps(S0=S0, v0=v0, r=r, k=k, theta=theta, sigma=sigma, rho=rho):

#     K_strike = np.linspace(round(S0-S0*0.3), round(S0+S0*0.3), 10)
#     Timeline = np.arange(0.1, 1.1, 0.1)

#     puts = np.zeros((len(Timeline), len(K_strike)))
#     calls = np.zeros((len(Timeline), len(K_strike)))

#     put_ivs = np.zeros((len(Timeline), len(K_strike)))
#     call_ivs = np.zeros((len(Timeline), len(K_strike)))


#     S_T = np.array([Heston_Sim(S0, v0, r, k, theta, sigma, rho, T) for T in Timeline])


#     for idB, T in enumerate(Timeline):        # GOOD CODE (i think)
#         for idA, k in enumerate(K_strike):

#             puts[idB,idA] = np.exp(-r*T)*np.mean(np.maximum(k-S_T[idB], 0))
#             calls[idB,idA] = np.exp(-r*T)*np.mean(np.maximum(S_T[idB]-k, 0))

#     # for idA, T in enumerate(Timeline):
#     #     put_ivs[idA, :] = implied_vol(puts[idA, :], S0, K_strike, T, r, flag='p', q=0, return_as='numpy', on_error='ignore')
#     #     call_ivs[idA, :] = implied_vol(calls[idA, :], S0, K_strike, T, r, flag='c', q=0, return_as='numpy')


#     put_ivs = implied_vol(puts[-1, :], S0, K_strike, T[-1], r, flag='p', q=0, return_as='numpy', on_error='ignore')


#     fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,6))
#     put_heatmap = ax1.imshow(put, cmap='RdYlGn', interpolation='nearest')
#     call_heatmap = ax2.imshow(call, cmap='RdYlGn', interpolation='nearest')
#     # Add color bar
#     fig.colorbar(put_heatmap, ax=ax1, label='Intensity')

#     # Add annotations
#     for i in range(10):
#         for j in range(10):
#             ax1.text(j, i, f'{put[i, j]:.2f}', ha='center', va='center', color='black')
#             ax2.text(j, i, f'{call[i, j]:.2f}', ha='center', va='center', color='black')

#     return put_ivs





# Main Page for Output Display
st.title("Heston Pricing Model")

# Table of Inputs
input_data = {
    "Initial Asset Price": [current_price],
    # "Strike Price": [strike],
    "Time to Maturity": [time_to_maturity],
    "Initial Volatility": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")








st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Currently working on updates... Stay tuned!")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")
