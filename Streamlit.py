import streamlit as st
import numpy as np 
from matplotlib import pyplot as plt 
import scipy as sp
import scipy.stats as ss
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function for Monte Carlo simulation of the Heston model
def Full_Heston_Sim(S0, v0, r, k, theta, sigma, rho, T, N_steps, N_sims):
    """
    Performs a Monte Carlo simulation of the Heston Model under risk-neutral dynamics to price vanilla options. 

    Args:   
            S0: Initial asset price (dollars)
            v0: Initial asset volatility ()
            r: Risk-free rate
            k (kappa): Velocity of mean reversion of the variance
            theta: Long-term variance mean
            sigma: Volatility of the variance (vol of vol)
            rho: Correlation factor between the asset price and variance wiener processes WS & Wv

    Output: 
            S: Asset price (type: Numpy array, Shape: (N_simulations, N_steps))
            v: Asset volatility (type: Numpy array, Shape: (N_simulations, N_steps))
            timeline: Simulation time array (type: Numpy array, size(N_steps), bounds(0, T))
    """

    # Initialize asset price and variance arrays, and set initial conditions
    S = np.zeros((N_sims, N_steps))
    v = np.zeros((N_sims, N_steps))
    v[:, 0], S[:,0] = v0, S0

    timeline = np.linspace(0, T, N_steps)   # Time array, size=N_steps
    dt = T/(N_steps-1)                      # Time step size

    mu = np.array([0, 0])       # Average value for WS & Wv. Values are 0 to represent the standard normal dist.
    cov = np.matrix([[1, rho],  # Covanriance matrix. Diagonal elements represent the variance of each sampling process
                    [rho, 1]])  # diagonal elements represent the correlation between the two factors
    
    # Creating (N_sims, N_steps) array for wiener process factors
    W = ss.multivariate_normal.rvs(mean=mu, cov=cov, size=(N_sims, N_steps - 1))    # N_steps-1 since v0 and S0 are accounted for. 
    WS = W[:, :, 0]  # Asset Brownian motion
    Wv = W[:, :, 1]  # Variance Brownian motion

    for j in range(N_sims):
        for idx, i in enumerate(timeline[1:]):
            v[j, idx+1] = np.maximum(v[j,idx] + k*(theta-v[j,idx])*dt + sigma*np.sqrt(v[j,idx]*dt) * Wv[j,idx], 0)
            S[j, idx+1] = S[j,idx] * np.exp( (r - 0.5*v[j,idx])*dt + np.sqrt(v[j,idx]*dt)*WS[j,idx] )

#     print(f"Final Variance: {np.mean(v[:,-1])} per unit time")
#     print(f"Final Mean Asset Price: ${np.mean(S[:,-1])}")

    return S, v, timeline

# Initial parameters
S0 = 100.0          
v0 = 0.06           
T = 1.0             
r = .02             
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
    r = st.number_input("Risk-Free Rate", value=r)
    rho = st.number_input("Wiener Correlation Factor", value=rho)

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
st.markdown("""The Heston Model is defined by set of stochastic differential equations (SDEs) describes how the price and volatility of an asset, security or stock could mature on the stock market
            over some time interval $T$. The instantaneous change in the price, and volatility of the asset at time $t$ ($dS_t$ and $dv_t$ respectively) are governed by 
            [Ornstein-Uhlenbeck processes](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) (Equations 1 and 2).""", unsafe_allow_html=True)

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

st.markdown("""Where the Wiener processes $dW_{S,t}$ and $dW_{v,t}$ (for asset price and volatility respectively) are Monte Carlo processes to govern the "random walk" motion of stocks.These two 
            Wiener processes are correlated by a factor $\\rho$, as detailed by Equation 3""", unsafe_allow_html=True)

## dWv*dWS = pdt
st.markdown("""
$$
dW_{S,t} \cdot dW_{v,t} = \\rho dt \quad (3)
$$
""", unsafe_allow_html=True)

st.markdown("""$S_t$ and $v_t$ are price and volatility of the asset at time $t$, and $dt$ is the difference in time between successive solutions. The risk-free rate $r$ (drift constant), 
            mean reversion rate of the volatility, long-term mean of the volatility, $\\theta$, and the volatility of the volatility \\sigma are simulation parameters that are set by the user (see right banner).
            """, unsafe_allow_html=True)

st.markdown("""This simulation relies on calculating the instantaneous asset volatility to update the price of the volatility in successive time steps using the Euler-discretized form of the
             Heston model (given initial conditions $S_0$, $v_0$).""", unsafe_allow_html=True)

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

st.markdown("""Where $W_v$ and $W_S$ are the wiener processes governed by a Monte Carlo approach following Equation 3.""", unsafe_allow_html=True)

st.header("Simulation Breakdown")
st.markdown(rf"""The plots below are the result $N_{{sims}}=${N_sims} simulations stepping through Equations 4 and 5 $N_{{steps}}=${N_steps} times over a $T={T}$ year long time-frame, 
            given the state of the parameters on the left banner.
            """, unsafe_allow_html=True)

# Run simulation
S, v, timeline = Full_Heston_Sim(S0=S0, v0=v0, r=r, k=k, theta=theta, sigma=sigma, rho=rho, T=T, N_steps=N_steps, N_sims=N_sims)

# Create Plotly subplot figure
fig = make_subplots(rows=1, cols=2, subplot_titles=('Asset Prices', 'Asset Volatility'))

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
    st.markdown(f"""<br><br>The effects of the correlation factor $\\rho$ can be observed by overlaying the price and volatility curves of a single simulation
                A negative correaltion $\\rho < 0$ causes the price and volatility to follow opposing trends, that is, when the volatility of the stock is high, 
                the asset price tends to fall, and vice versa. Conversely, when $\\rho > 0$, the $S_t$ and $v_t$ curves tend to follow each other.""", unsafe_allow_html=True)
    
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


# Beginning of the next section. Re-establish the variables and run heston agian. 

# Initial parameters
S0 = 100.0          
v0 = 0.06           
T = 1.0             
r = .02             
rho = -0.7          

k = 3               
theta = 0.04        
sigma = 0.6         
N_sims = 20
N_steps = 100

S, v, timeline = Full_Heston_Sim(S0=S0, v0=v0, r=r, k=k, theta=theta, sigma=sigma, rho=rho, T=T, N_steps=100, N_sims=500)

st.header("Pricing the Discounted Payoff for Put and Call Options")
st.markdown("For further analysis we can determine the simulated price of the stock at maturity by calculating the mean asset price at maturity", unsafe_allow_html=True)

# st.markdown("""
# $$
# S_T = \frac{1}{N} \sum_{i=1}^{N} S_T^i
# $$
# """, unsafe_allow_html=True)

st.markdown("""
    $$
    S_T = \\frac{1}{N} \sum_{i=1}^{N} S_T^i
    $$
""")

st.markdown("where $N$ is the number of simualted paths.", unsafe_allow_html=True)





st.info("This page is a work in progress! I am working on adding more depth to this analysis. Stay tuned!")