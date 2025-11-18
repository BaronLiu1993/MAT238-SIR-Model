import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


infectious_period = 11           
N_population = 11_770_000        
beta_estimate = 2.002  
sigma_estimate = 0.1          

df = pd.read_csv("g.csv")
df['Date'] = pd.to_datetime(df['Date'])


df['new_cases'] = df['cum_cases'].diff().fillna(0)
df['new_deaths'] = df['cum_deaths'].diff().fillna(0)

df['active_cases'] = df['cum_cases'] - df['cum_cases'].shift(infectious_period).fillna(0)
df['recovered'] = df['cum_cases'].shift(infectious_period).fillna(0) - df['cum_deaths']
df['removed'] = df['recovered'] + df['cum_deaths']

df['I'] = df['active_cases'].clip(lower=0)
df['R'] = df['removed'].clip(lower=0)
df['S'] = (N_population - df['I'] - df['R']).clip(lower=0)

df['s'] = df['S']
df['i'] = df['I']
df['r'] = df['R']

nu = 1 / infectious_period
df['di_dt'] = df['i'].diff().fillna(0)
R0 = beta_estimate / nu


T_days = len(df) - 1    
dt = 0.1               
num_steps = int(T_days / dt) + 1

t = np.linspace(0, T_days, num_steps)

S = np.zeros(num_steps)
E = np.zeros(num_steps) 
I = np.zeros(num_steps)  
R = np.zeros(num_steps)

# Initial conditions
S[0] = N_population - 1
I[0] = 1
E[0] = 0
R[0] = 0

for k in range(1, num_steps):
    dS = (-beta_estimate * S[k - 1] * I[k - 1]) / N_population
    dE = (beta_estimate * S[ k - 1] * I[k - 1]) / N_population - sigma_estimate * E[k - 1]
    dI = (beta_estimate * S[k - 1] * I[k - 1]) / N_population - nu * I[k - 1]
    dR = nu * I[k - 1]

    S[k] = S[k - 1] + dS * dt
    I[k] = I[k - 1] + dI * dt
    E[k] = E[k - 1] + dE * dt # Exposed Model
    R[k] = R[k - 1] + dR * dt

start_date = df['Date'].iloc[0]
date_fine = start_date + pd.to_timedelta(t, unit="D")


plt.figure(figsize=(12, 6))

plt.plot(date_fine, S, linewidth=2, label='Susceptible $S(t)$ (model)')
plt.plot(date_fine, I, linewidth=2, label='Infected $I(t)$ (model)')
plt.plot(date_fine, E, linewidth=2, label='Exposed $E(t)$ (model)')
plt.plot(date_fine, R, linewidth=2, label='Removed $R(t)$ (model)')

plt.title(
    "Guinea Ebola Outbreak – SIR Model\n"
    f"SIR parameters: β = {beta_estimate:.3f}, ν = {nu:.3f} (1/day), "
    f"$R_0$ = {R0:.2f}, Population N = {N_population:,}"
)

plt.xlabel("Date")
plt.ylabel("Number of individuals")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))       
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))       
plt.gcf().autofmt_xdate()

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
