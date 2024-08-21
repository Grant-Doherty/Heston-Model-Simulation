import streamlit as st
import numpy as np 
from matplotlib import pyplot as plt 
import scipy as sp
import scipy.stats as ss
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import importlib
import inspect

# Import the utils module
utils = importlib.import_module('utils')

# Loop over all members of the utils module
for name, func in inspect.getmembers(utils, inspect.isfunction):
    # Assign the function to a global variable with the same name
    globals()[name] = func

# Initial parameters
S0 = 100.0          
v0 = 0.06           
T = 1.0             
r = .15             
rho = -0.7          

k = 3               
theta = 0.04        
sigma = 0.6         
N_sims = 20
N_steps = 100

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

    S0 = st.number_input("Initial Asset Price $S_0$ (dollars)", value=S0)
    v0 = st.number_input("Initial Volatility $v_0$ (years$^{-1}$)", value=v0)
    T = st.number_input("Time to Maturity $T$ (years)", value=T)
    r = st.number_input("Risk-Free Rate $r$", value=r)
    rho = st.number_input("Wiener Correlation Factor $\\rho$", value=rho)
    K = st.number_input("Strike Price $K$", value=S0*1.3)

    st.markdown("<p style='font-size: 20px;'><strong>Fixed Parameters</strong></p>",
    unsafe_allow_html=True)

    st.markdown(f"Velocity of Mean Reversion: κ={k}", unsafe_allow_html=True)
    st.markdown(f"Long-Term Mean of the Volatility: θ={theta}", unsafe_allow_html=True)
    st.markdown(f"Volatility of the volatility: σ={sigma}", unsafe_allow_html=True)
    st.markdown(f"Number of Simulations: N_sims={N_sims}", unsafe_allow_html=True)
    st.markdown(f"Number of Steps per Simulation: N_steps={N_steps}", unsafe_allow_html=True)

# st.markdown("")
# st.title("Options Price - Interactive Heatmap")
# st.info("Below is a plot of the simulation over the timeframe. This is a work in progress! I intend on providing more details on the meaningful insight in this simulation.")

st.markdown("")
st.header("Background Information")
st.markdown("""The Heston Model is defined by a set of stochastic differential equations (SDEs) that describe how the price and volatility of an asset could mature on the stock market
            over some time interval $T$. The instantaneous change in the price, and volatility of the asset at time $t$ ($dS_t$ and $dv_t$, respectively) are governed by 
            [Ornstein-Uhlenbeck processes](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) (Equations $1$ and $2$)""", unsafe_allow_html=True)

## dS_t and dv_t 
st.markdown("""
            
$$
dS_t = rS_tdt+\sqrt{v_t}S_tdW_{S,t} \quad (1)
$$
""", unsafe_allow_html=True)
st.markdown("""
$$
dv_t = \kappa (\\theta - v_t)dt + \sigma \sqrt{v_t}dW_{v,t} \quad (2)
$$
""", unsafe_allow_html=True)

st.markdown("""where the Wiener processes $dW_{S,t}$ and $dW_{v,t}$ (for asset price and volatility, respectively) are Monte Carlo processes to govern the "random" motion of stocks. These two 
            Wiener processes are correlated by $\\rho$, as detailed by Equation 3""", unsafe_allow_html=True)

## dWv*dWS = pdt
st.markdown("""
$$
dW_{S,t} \cdot dW_{v,t} = \\rho dt \quad (3)
$$
""", unsafe_allow_html=True)

st.markdown("""$S_t$ and $v_t$ are price and volatility of the asset at time $t$, and $dt$ is an infinitessimal time element. The risk-free rate $r$ (drift constant), 
            mean reversion rate of the volatility $\\kappa$, long-term mean of the volatility $\\theta$, and the volatility of the volatility $\\sigma$ are fixed simulation parameters
            (see right banner).
            """, unsafe_allow_html=True)

st.markdown("""This simulation relies on calculating the instantaneous asset volatility to update the price of the volatility in successive time steps using the Euler-discretized form of the
             Heston model""", unsafe_allow_html=True)

## v_{t+1} and S_{t+1}
st.markdown("""   
            $$
            v_{t+1} = v_t + \kappa (\\theta - v_t)dt + \sigma \sqrt{v_tdt}W_v \quad(4)
            $$
""", unsafe_allow_html=True)
st.markdown("""   
            $$
            S_{t+1} = S_t \ exp \left\{ (r - \\frac{1}{2}v_t)dt + \sqrt{v_tdt}W_s \\right\}  \quad (5) 
            $$
""", unsafe_allow_html=True)

st.markdown("""given initial conditions $S_0$, $v_0$. $W_v$ and $W_S$ are the correlated wiener processes that are computed using a Monte Carlo approach.""", unsafe_allow_html=True)

st.header("Simulation Breakdown")
st.markdown(rf"""The plots below are the result $N_{{sims}}={N_sims}$ simulations stepping through Equations $4$ and $5$ $N_{{steps}}={N_steps}$ times over a $T={T}$ year long time-frame, 
            given the state of the parameters on the left banner.
            """, unsafe_allow_html=True)

# Run simulation
S, v, timeline = Heston_Sim(S0=S0, v0=v0, r=r, k=k, theta=theta, sigma=sigma, rho=rho, T=T, N_steps=N_steps, N_sims=N_sims)

# Create Plotly subplot figure
fig = make_subplots(rows=1, cols=2, subplot_titles=('Asset Price', 'Asset Volatility'))

for i in range(N_sims):
                fig.add_trace(
                    go.Scatter(x=timeline, y=S[i, :], mode='lines', name=f'Sim {i+1}', showlegend=False),
                    row=1, col=1
)

                fig.add_trace(
                    go.Scatter(x=timeline, y=v[i, :], mode='lines', name=f'Sim {i+1}', showlegend=False),
                    row=1, col=2
)

# Update layout
fig.update_layout(
                title="Monte Carlo Simulation Results",
                width=1300,
                height=500,
                xaxis_title='Time (Years)',
                yaxis_title='Asset Price',
                xaxis2_title='Time (Years)',
                yaxis2_title='Asset Volatility (%/Year)'
)

# Display Plotly chart in Streamlit
st.plotly_chart(fig)        


st.subheader(rf"""How $\rho$ contributes""")
# Set up left and right side of the webpage
col1, col2 = st.columns([1,1], gap="small")

# On the left side...
with col1:
    # Info on the wiener correlation factor rho. 
    st.markdown(f"""<br><br>The effects of the correlation factor $\\rho$ can be observed by overlaying the price and volatility curves of a single simulation.
                A negative correaltion ($\\rho < 0$) causes the price and volatility to follow opposing trends, that is, when the volatility of the stock is high, 
                the asset price tends to fall, and vice versa. Conversely, a positive correlation ($\\rho > 0$) causes the $S_t$ and $v_t$ curves to follow each other. 
                Negative correlations are typically observed on the stock market due to the leverage effect and risk aversion.""", unsafe_allow_html=True)
    
    st.markdown(f"""**Try for yourself:** Adjust the $\\rho$ parameter on the left banner between positive and negative values to see how it affects the correlation between $S_t$ and $v_t$. \
                Note that larger (more positive or more negative) values of $\\rho$ will intensify the correlation.""", unsafe_allow_html=True)

    st.markdown(f"""**Note**:
- Correlation effects may be more pronounced in some simulations than in others. You can re-run the simulation by pressing the 'r' key or by selecting "Rerun" in the top right corner.
- The graphs are interactive! Zoom in by clicking and dragging, and try other tools using the banner along the top right.
""", unsafe_allow_html=True)


# On the right side...
with col2:
    # 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline, y=S[0,:], mode='lines', name='Asset Price', yaxis='y1'))
    fig.add_trace(go.Scatter(x=timeline, y=v[0,:], mode='lines', name='Asset Volatility', yaxis='y2'))

    # Figure layout and customization
    fig.update_layout(title='Overlayed Price and Volatility Curves of a Single Simulation',
                        width=900,
                        height=500,
                        xaxis_title='Time (Years)',
                        yaxis_title='',
                    
                    yaxis=dict(
                        title='Price',
                        showgrid=False,
                    ),

                    yaxis2=dict(
                        title='Volatility',
                        overlaying='y',
                        side='right',
                        showgrid=False,
                    ),

                    xaxis=dict(title='Time',
                    showgrid=False)

)

    st.plotly_chart(fig)

# ------------------------------------------------------------------------------------------------------------------------------------------------- #

st.header("Options")
# Explaining what options are
st.markdown("""Options are financial contracts that give the owner (the holder of the option) the right, but not the obligation, to buy or sell a specific quantity of an
             underlying asset at a specific strike price, on or before a specified date, depending on the type of option.
             Options can be sold as a call option, or a put option:

- **Call Option:** gives the holder the right to buy an underlying asset at a specific price (strike price).
- **Put Option:** gives the holder the right to sell an underlying asset at a specific price.

There are two common types of option contracts: American, and European options. American options can be exercised at any time up to the date of expiration, while \
            European options can only be exercised at the expiration date. Check out this [great YouTube video](https://youtu.be/VJgHkAqohbU?t=174) by The Plain Bagel for more information on options.
""", unsafe_allow_html=True)

st.subheader("Pricing Options")
# Why we need to price options.
st.markdown("""Unlike stocks, the price of an option is not strictly defined on a market, and institutions rely on models to price options.
            This simulaton can be used as a tool to price European options. In this simulation, the average call/put payoff can be calculated by averaging the individual call/put payoffs.
""", unsafe_allow_html=True)

# EQ Av Call/Put Payoff
st.markdown("""
    $$
    \\text{Average Call Payoff} = \\frac{1}{N} \sum_{i=1}^{N} \ S_T^i \ \\text{max}(S_T^i-K,0) \qquad (6)
    $$
""") 
st.markdown("""
    $$
    \\text{Average Put Payoff} = \\frac{1}{N} \sum_{i=1}^{N} \ S_T^i \ \\text{max}(K-S_T^i,0) \qquad (7)
    $$
""")

st.markdown("""where $S_T^i$ is the $i^{\\text{th}}$ asset price at maturiton, $K$ is the strike price, and $N$ is the number of simulations. Since options are priced at the present value, 
            the average payoffs should be discounted using the risk-free rate $r$, leading to final equations for the price of call and put options:
""", unsafe_allow_html=True)

# EQ Theoretical Call/Put Price
st.markdown("""
    $$
    \\text{Call Price} = e^{-rT} \\times \\text{Average Call Payoff} \qquad (8)
    $$
""", unsafe_allow_html=True)
st.markdown("""
    $$
    \\text{Put Price} = e^{-rT} \\times \\text{Average Put Payoff} \qquad (9)
    $$
""", unsafe_allow_html=True)

N_steps_const = 250
N_sims_const = 500

st.markdown(rf"""
    The table below represents the simulated value of the call and put given the state of the parameters set to the left. Since the Wiener processes draw from standard normal distributions and
             prices are dependent on averages, the law of large numbers applies and it is desirable to have a large number of simulations to get accurate pricing. The following pricing is calculated using 
            $N_{{sims}}={N_sims_const}$ and $N_{{steps}}={N_steps_const}$.
""", unsafe_allow_html=True)

st.markdown(rf"""**NOTE:** This page compiles linearly and the information below must be computed every time variables are updated. Expect computation time delays (no longer than 30s).
""", unsafe_allow_html=True)

# Beginning of the next section. Re-establish the variables and run heston agian. 
N_steps_const = 252
N_sims_const = 500
S_T,_,_ = Heston_Sim(S0=S0, v0=v0, r=r, k=k, theta=theta, sigma=sigma, rho=rho, T=T, N_steps=N_steps_const, N_sims=N_sims_const, Final=True)
Call_price, Put_price = Payoff(S_T,r=r, T=T, K=K)


df = pd.DataFrame({
    "Option Type": ["Mean Asset Price at Maturity", "Call Option Price", "Put Option Price"],
    "Price": [np.mean(S_T), Call_price, Put_price]
})

col1, col2 = st.columns([1,1], gap="small")
with col1:
    st.dataframe(df, use_container_width=True)

st.info("This page is a work in progress! I am working on adding more depth to this analysis. Stay tuned!")

st.header("References")

st.markdown("""

1. **Rouah, F.** *Euler and Milstein Discretization*. [Link](https://frouah.com/finance%20notes/Euler%20and%20Milstein%20Discretization.pdf).
   
2. **Heston, S.** (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*.  \
            The Review of Financial Studies. [Link](A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options).

3. **Haugh, M.** (2016). *The Black-Scholes Model*. Foundations of Financial Engineering. [Link](https://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf).

""")