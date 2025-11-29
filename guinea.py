import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Parameters
infectious_period = 11           # days
N_population = 11_770_000        # total population
beta_estimate = 0.5              # transmission rate β
sigma_estimate = 0.100           # incubation rate σ
nu = 1 / infectious_period       # recovery rate ν
R0_basic = beta_estimate / nu    # basic reproduction number R0

# Load and process data
df = pd.read_csv("g.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Daily new cases / deaths
df["new_cases"] = df["cum_cases"].diff().fillna(0)
df["new_deaths"] = df["cum_deaths"].diff().fillna(0)

# Very rough active / removed reconstruction
df["active_cases"] = df["cum_cases"] - df["cum_cases"].shift(infectious_period).fillna(0)
df["recovered"] = df["cum_cases"].shift(infectious_period).fillna(0) - df["cum_deaths"]
df["removed"] = df["recovered"] + df["cum_deaths"]

df["I"] = df["active_cases"].clip(lower=0)
df["R"] = df["removed"].clip(lower=0)
df["S"] = (N_population - df["I"] - df["R"]).clip(lower=0)

df["I_data"] = (
    df["new_cases"]
    .rolling(infectious_period, min_periods=1)
    .sum()
    .clip(lower=0)
)

#SEIR model integration on daily grid (Euler)
T_days = len(df) - 1
dt = 1.0  # day
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
E[0] = 0.0
I[0] = I0
R[0] = R0_data

for k in range(1, num_steps):
    S_prev, E_prev, I_prev, R_prev = S[k-1], E[k-1], I[k-1], R[k-1]

    dS = -beta_estimate * S_prev * I_prev / N_population
    dE = beta_estimate * S_prev * I_prev / N_population - sigma_estimate * E_prev
    dI = sigma_estimate * E_prev - nu * I_prev
    dR = nu * I_prev

    S[k] = S_prev + dS * dt
    E[k] = E_prev + dE * dt
    I[k] = I_prev + dI * dt
    R[k] = R_prev + dR * dt

# Calendar dates for the model
start_date = df["Date"].iloc[0]
date_model = start_date + pd.to_timedelta(t, unit="D")

def thousands_formatter(x, pos):
    return f"{x / 1000:.1f}"

# Figure 1: SEIR + SIR trajectories (model only)
plt.figure(figsize=(12, 6))

plt.plot(date_model, S, linewidth=2, label="Susceptible S(t)", color="tab:blue")
plt.plot(date_model, E, linewidth=2, label="Exposed E(t)", color="tab:green")
plt.plot(date_model, I, linewidth=2, label="Infected I(t)", color="orange")
plt.plot(date_model, R, linewidth=2, label="Removed R(t)", color="red")

plt.title(
    "Guinea Ebola Outbreak – SEIR Model\n"
    f"β = {beta_estimate:.3f}, ν = {nu:.3f} (1/day), "
    f"σ = {sigma_estimate:.3f}, $R_0$ ≈ {R0_basic:.2f}, N = {N_population:,}"
)
plt.xlabel("Date")
plt.ylabel("Number of individuals (thousands)")

ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gcf().autofmt_xdate()

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# Figure 2: Infected I(t) model vs. constructed infectious data
plt.figure(figsize=(12, 5))

plt.plot(date_model, I, linewidth=2, label="Infected I(t) (model)")
plt.scatter(df["Date"], df["I_data"], s=25, color="red", alpha=0.8, label="Infectious (data)")

ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gcf().autofmt_xdate()

plt.title("Guinea Infected I(t): Model vs. Constructed Infectious Data")
plt.xlabel("Date")
plt.ylabel("Infected individuals (thousands)")

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
