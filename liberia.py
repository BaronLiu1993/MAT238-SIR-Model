import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Parameters
infectious_period = 11           # Infectious period (days)
N_population = 4_660_000        # Total population
beta_estimate = 1.365           # Transmission rate (β)
sigma_estimate = 0.111          # Incubation rate (σ)
nu = 1 / infectious_period      # Recovery rate (ν)
R0_basic = beta_estimate / nu   # Basic reproduction number (R0)

# Load and process the data
df = pd.read_csv("l.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Calculate daily new cases and deaths
df["new_cases"] = df["cum_cases"].diff().fillna(0)
df["new_deaths"] = df["cum_deaths"].diff().fillna(0)

# Calculate active cases, recovered, and removed (total)
df["active_cases"] = df["cum_cases"] - df["cum_cases"].shift(infectious_period).fillna(0)
df["recovered"] = df["cum_cases"].shift(infectious_period).fillna(0) - df["cum_deaths"]
df["removed"] = df["recovered"] + df["cum_deaths"]

df["I"] = df["active_cases"].clip(lower=0)
df["R"] = df["removed"].clip(lower=0)
df["S"] = (N_population - df["I"] - df["R"]).clip(lower=0)

df["I_data"] = df["new_cases"].rolling(infectious_period, min_periods=1).sum().clip(lower=0)

# 3. SEIR model integration using Euler's method
T_days = len(df) - 1  
dt = 0.1               
num_steps = int(T_days / dt) + 1

t = np.linspace(0, T_days, num_steps)

S = np.zeros(num_steps)
E = np.zeros(num_steps) 
I = np.zeros(num_steps)
R = np.zeros(num_steps)

I0 = df["I"].iloc[0]
R0_data = df["R"].iloc[0]
S0 = N_population - I0 - R0_data

S[0] = S0
I[0] = I0
R[0] = R0_data
E[0] = 0

# Euler integration to solve the SEIR system
for k in range(1, num_steps):
    dS = -beta_estimate * S[k - 1] * I[k - 1] / N_population
    dE = beta_estimate * S[k - 1] * I[k - 1] / N_population - sigma_estimate * E[k - 1]
    dI = sigma_estimate * E[k - 1] - nu * I[k - 1]
    dR = nu * I[k - 1]

    S[k] = S[k - 1] + dS * dt
    E[k] = E[k - 1] + dE * dt
    I[k] = I[k - 1] + dI * dt
    R[k] = R[k - 1] + dR * dt

start_date = df["Date"].iloc[0]
date_fine = start_date + pd.to_timedelta(t, unit="D")

# Plotting Functionality (Clean and Clear)

def thousands_formatter(x, pos):
    return f'{x / 1000:.1f}'

# Figure 1: SEIR Model (with Susceptible, Infected, Removed)
plt.figure(figsize=(12, 6))

plt.plot(date_fine, S / 1000, linewidth=2, label='Susceptible S(t)', color="tab:blue")
plt.plot(date_fine, I / 1000, linewidth=2, label='Infected I(t)', color="orange")
plt.plot(date_fine, R / 1000, linewidth=2, label='Removed R(t)', color="red")

plt.title(
    "Liberia Ebola Outbreak – SEIR Model\n"
    f"β = {beta_estimate:.3f}, ν = {nu:.3f} (1/day), "
    f"σ = {sigma_estimate:.3f}, $R_0$ = {R0_basic:.2f}, N = {N_population:,}"
)
plt.xlabel("Date")
plt.ylabel("Number of individuals (thousands)")

ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gcf().autofmt_xdate()

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# Figure 2: Infected I(t) Model vs Raw Data (Daily)
plt.figure(figsize=(10, 5))

plt.plot(date_fine, I, linewidth=2, label="Infected I(t) (model)")
plt.scatter(df["Date"], df["I_data"], s=25, color="red", alpha=0.8, label="Raw (data)")

ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45, ha="right")

plt.title("Liberia I(t) Model Against Raw Infectious Count from Data")
plt.xlabel("Date")
plt.ylabel("Infected Individuals (thousands)")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

