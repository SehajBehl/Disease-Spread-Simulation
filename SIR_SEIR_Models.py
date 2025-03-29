import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------------------------------------
# Load the COVID data and filter for Canada in 2020
data = pd.read_csv('Covid_Data.csv')
# Assuming the CSV has columns 'Country/Region' and 'Date'
data = data[data['Country/Region'] == 'Canada']
data['Date'] = pd.to_datetime(data['Date'])
data = data[data['Date'].dt.year == 2020]
print("Data head:")
print(data.head())

# Use only the first 'days' rows for simulation validation if needed
days = 160  # simulation period
date_range = data['Date'].iloc[:days]  # Use these dates for x-axis

# Total population for simulation (adjust as needed)
N = 1000000  

# -----------------------------------------------------------------------------
# SIR Model Definitions
# -----------------------------------------------------------------------------
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def run_sir_simulation(beta, gamma, I0, R0, days):
    S0 = N - I0 - R0
    t = np.linspace(0, days, days)
    y0 = S0, I0, R0
    ret = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = ret.T
    return t, S, I, R

# -----------------------------------------------------------------------------
# SEIR Model Definitions
# -----------------------------------------------------------------------------
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def run_seir_simulation(beta, sigma, gamma, I0, E0, R0, days):
    S0 = N - I0 - E0 - R0
    t = np.linspace(0, days, days)
    y0 = S0, E0, I0, R0
    ret = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
    S, E, I, R = ret.T
    return t, S, E, I, R

# -----------------------------------------------------------------------------
# Monte Carlo Simulation Function
# -----------------------------------------------------------------------------
def monte_carlo_simulation(model_func, runs, model_params, initial_conditions, days):
    """
    Run multiple simulation runs (Monte Carlo) for either the SIR or SEIR model.
    """
    results = []
    for run in range(runs):
        # Randomly sample parameters within specified ranges.
        sampled_params = {param: random.uniform(*val_range) for param, val_range in model_params.items()}
        if model_func == run_sir_simulation:
            t, S, I, R = model_func(sampled_params['beta'], sampled_params['gamma'],
                                    initial_conditions['I0'], initial_conditions['R0'], days)
            results.append({'t': t, 'S': S, 'I': I, 'R': R, 'params': sampled_params})
        elif model_func == run_seir_simulation:
            t, S, E, I, R = model_func(sampled_params['beta'], sampled_params['sigma'], sampled_params['gamma'],
                                       initial_conditions['I0'], initial_conditions.get('E0', 0), initial_conditions['R0'], days)
            results.append({'t': t, 'S': S, 'E': E, 'I': I, 'R': R, 'params': sampled_params})
    return results

# -----------------------------------------------------------------------------
# Parameters and Initial Conditions
# -----------------------------------------------------------------------------
# Initial conditions for both models
initial_conditions_sir = {
    'I0': 100,
    'R0': 0
}
initial_conditions_seir = {
    'I0': 100,
    'E0': 50,  # assuming some individuals are in the incubation phase
    'R0': 0
}

# Parameter ranges for Monte Carlo sampling:
sir_params_range = {
    'beta': (0.1, 0.5),
    'gamma': (0.05, 0.2)
}
seir_params_range = {
    'beta': (0.1, 0.5),
    'sigma': (1/7, 1/3),  # incubation period from 3 to 7 days
    'gamma': (0.05, 0.2)
}

# -----------------------------------------------------------------------------
# Run Simulations: SIR and SEIR with Monte Carlo
# -----------------------------------------------------------------------------
runs = 50
sir_simulations = monte_carlo_simulation(run_sir_simulation, runs, sir_params_range, initial_conditions_sir, days)
seir_simulations = monte_carlo_simulation(run_seir_simulation, runs, seir_params_range, initial_conditions_seir, days)

# -----------------------------------------------------------------------------
# Intervention Strategies Example (Deterministic SEIR)
# -----------------------------------------------------------------------------
def apply_intervention(initial_S, vaccination_rate, beta, distancing_reduction):
    # Adjust susceptible count due to vaccination and beta due to social distancing.
    S_new = initial_S * (1 - vaccination_rate)
    beta_new = beta * (1 - distancing_reduction)
    return S_new, beta_new

# Baseline parameters (example values)
baseline_beta = 0.3
baseline_gamma = 0.1
baseline_sigma = 1/5  # incubation period of 5 days

# Scenario A: 30% vaccination, 20% beta reduction (moderate distancing)
vaccination_rate_A = 0.30
distancing_reduction_A = 0.20
# Scenario B: 10% vaccination, no beta reduction
vaccination_rate_B = 0.10
distancing_reduction_B = 0.0

# Initial susceptible count from SEIR initial conditions
S0 = N - initial_conditions_seir['I0'] - initial_conditions_seir.get('E0', 0) - initial_conditions_seir['R0']
S0_A, beta_A = apply_intervention(S0, vaccination_rate_A, baseline_beta, distancing_reduction_A)
S0_B, beta_B = apply_intervention(S0, vaccination_rate_B, baseline_beta, distancing_reduction_B)

def run_deterministic_seir(S0, beta, sigma, gamma, I0, E0, R0, days):
    t = np.linspace(0, days, days)
    y0 = S0, E0, I0, R0
    ret = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
    S, E, I, R = ret.T
    return t, S, E, I, R

t_A, S_A, E_A, I_A, R_A = run_deterministic_seir(S0_A, beta_A, baseline_sigma, baseline_gamma,
                                                  initial_conditions_seir['I0'], initial_conditions_seir.get('E0', 0),
                                                  initial_conditions_seir['R0'], days)
t_B, S_B, E_B, I_B, R_B = run_deterministic_seir(S0_B, beta_B, baseline_sigma, baseline_gamma,
                                                  initial_conditions_seir['I0'], initial_conditions_seir.get('E0', 0),
                                                  initial_conditions_seir['R0'], days)

# -----------------------------------------------------------------------------
# Model Validation (Real vs Simulated Data)
# -----------------------------------------------------------------------------
# Assume our CSV contains a column 'Confirmed' representing daily confirmed infected counts.
simulated_infected = sir_simulations[0]['I'][:days]
real_infected = data['Confirmed'].values[:days]  # now both arrays have length 'days'
rmse = np.sqrt(mean_squared_error(real_infected, simulated_infected))
print(f"RMSE between simulated and real infected counts (SIR model): {rmse:.2f}")

# -----------------------------------------------------------------------------
# Visualization: All Plots in One Figure with Dates on X-Axis
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# Subplot 1: SIR Monte Carlo Simulations (Infected)
for sim in sir_simulations[:5]:
    axs[0, 0].plot(date_range, sim['I'][:len(date_range)], alpha=0.6,
                   label=f"β={sim['params']['beta']:.2f}, γ={sim['params']['gamma']:.2f}")
axs[0, 0].set_title('SIR Monte Carlo Simulations (Infected)')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Infected Individuals')
axs[0, 0].legend(fontsize=8)
axs[0, 0].grid(True)

# Subplot 2: SEIR Monte Carlo Simulations (Infected)
for sim in seir_simulations[:5]:
    axs[0, 1].plot(date_range, sim['I'][:len(date_range)], alpha=0.6,
                   label=f"β={sim['params']['beta']:.2f}, σ={sim['params']['sigma']:.2f}, γ={sim['params']['gamma']:.2f}")
axs[0, 1].set_title('SEIR Monte Carlo Simulations (Infected)')
axs[0, 1].set_xlabel('Date')
axs[0, 1].set_ylabel('Infected Individuals')
axs[0, 1].legend(fontsize=8)
axs[0, 1].grid(True)

# Subplot 3: Intervention Scenario Comparison (Deterministic SEIR)
axs[1, 0].plot(date_range, I_A[:len(date_range)], label='Scenario A\n(30% vaccination, 20% distancing)')
axs[1, 0].plot(date_range, I_B[:len(date_range)], label='Scenario B\n(10% vaccination, 0% distancing)')
axs[1, 0].set_title('SEIR: Intervention Strategies Comparison')
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel('Infected Individuals')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Subplot 4: Model Validation (Real vs Simulated Infected)
axs[1, 1].plot(date_range, real_infected[:len(date_range)], label='Real Infected Data')
axs[1, 1].plot(date_range, simulated_infected[:len(date_range)], label='SIR Simulation')
axs[1, 1].set_title(f'Model Validation (RMSE: {rmse:.2f})')
axs[1, 1].set_xlabel('Date')
axs[1, 1].set_ylabel('Infected Individuals')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# End of Simulation Script
# -----------------------------------------------------------------------------
