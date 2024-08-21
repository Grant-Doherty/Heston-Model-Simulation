import streamlit as st
import numpy as np 
from matplotlib import pyplot as plt 
import scipy as sp
import scipy.stats as ss
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

func_names = [
    'Heston_Sim',
    'Payoff'
]

def Heston_Sim(S0, v0, r, k, theta, sigma, rho, T, N_steps=100, N_sims=1000, Final=False):

    """
    Performs a Monte Carlo simulation of the Heston Model under risk-neutral dynamics. 

    Args:   
            S0: Initial asset price (dollars)
            v0: Initial asset volatility ()
            r: Risk-free rate
            k (kappa): Velocity of mean reversion of the variance
            theta: Long-term variance mean
            sigma: Volatility of the variance (vol of vol)
            rho: Correlation factor between the asset price and variance wiener processes WS & Wv
            T: Time to maturity (years)
            N_steps: # steps to maturity
            N_sims: # of simulated paths

    Output: 
            if Final=True:
                v_T: Sim's var at maturity (type: Numpy array, Shape: (N_simulations))
                S_T: Sim's asset price at maturity (type: Numpy array, Shape: (N_simulations))
                timeline: Simulation time array (type: Numpy array, size(N_steps), bounds(0, T))

            else ((Final=False)):
                S: Sim's asset price at array (type: Numpy array, Shape: (N_simulations, N_steps))
                v: Sim's var array (type: Numpy array, Shape: (N_simulations, N_steps))
                timeline: Simulation time array (type: Numpy array, size(N_steps), bounds(0, T))
    """

    # Initialize variance and asset price arrays, and set initial conditions
    v = np.zeros((N_sims, N_steps))
    S = np.zeros((N_sims, N_steps))
    v[:, 0], S[:,0] = v0, S0

    timeline = np.linspace(0, T, N_steps)
    dt = T/(N_steps-1)

    mu = np.array([0, 0])
    cov = np.matrix([[1, rho],
                    [rho, 1]])
    W = ss.multivariate_normal.rvs(mean=mu, cov=cov, size=(N_sims, N_steps - 1))    # N_steps-1 since v0 and S0 are accounted for. 
    WS = W[:, :, 0]  # Stock Brownian motion:     W_1
    Wv = W[:, :, 1]  # Variance Brownian motion:  W_2

    for j in range(N_sims):
        for idx, i in enumerate(timeline[1:]):
            v[j, idx+1] = np.maximum(v[j,idx] + k*(theta-v[j,idx])*dt + sigma*np.sqrt(v[j,idx]*dt) * Wv[j,idx], 0)
            S[j, idx+1] = S[j,idx] * np.exp( (r - 0.5*v[j,idx])*dt + np.sqrt(v[j,idx]*dt)*WS[j,idx] )

    if Final:
        return S[:,-1], v[:,-1], timeline
    
    else:
        return S, v, timeline


def Payoff(S, r, T, K):
    P_call = np.exp(-r*T) * np.mean(np.maximum(S-K,0))
    P_put = np.exp(-r*T) * np.mean(np.maximum(K-S,0))
    return P_call, P_put






