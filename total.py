import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set larger font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13
})

# Liberia
infectious_period_l = 11
N_population_l = 4_660_000
beta_estimate_l = 1.365
sigma_estimate_l = 0.1

# Sierra Leone
infectious_period_sl = 11
N_population_sl = 7_040_000
beta_estimate_sl = 1.547
sigma_estimate_sl = 0.1

# Guinea
infectious_period_g = 11
N_population_g = 11_770_000
beta_estimate_g = 2.002
sigma_estimate_g = 0.1

# Load data (make sure these CSVs have Date, cum_cases, cum_deaths)
df_l = pd.read_csv("l.csv")
df_sl = pd.read_csv("sl.csv")
df_g = pd.read_csv("g.csv")

def simulate_country(df, infectious_period, N_population, beta_estimate, sigma_estimate):
    """
    Run the SEIR-style model for one country and return (date_fine, I).
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Duration and time step
    T_days = len(df) - 1
    dt = 0.1
    num_steps = int(T_days / dt) + 1

    t = np.linspace(0, T_days, num_steps)

    # Arrays
    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)

    # Initial conditions (1 infected person, rest susceptible)
    S[0] = N_population - 1
    I[0] = 1
    R[0] = 0
    E[0] = 0

    nu = 1 / infectious_period

    # Time stepping
    for k in range(1, num_steps):
        dS = -beta_estimate * S[k - 1] * I[k - 1] / N_population
        dE = (beta_estimate * S[k - 1] * I[k - 1]) / N_population - sigma_estimate * E[k - 1]
        dI = sigma_estimate * E[k - 1] - nu * I[k - 1]
        dR = nu * I[k - 1]

        S[k] = S[k - 1] + dS * dt
        E[k] = E[k - 1] + dE * dt
        I[k] = I[k - 1] + dI * dt
        R[k] = R[k - 1] + dR * dt

    start_date = df["Date"].iloc[0]
    date_fine = start_date + pd.to_timedelta(t, unit="D")

    return date_fine, I, beta_estimate, nu, N_population

# Simulate each country
liberia_dates, liberia_I, beta_l, nu_l, N_l = simulate_country(
    df_l, infectious_period_l, N_population_l, beta_estimate_l, sigma_estimate_l
)
sl_dates, sl_I, beta_sl, nu_sl, N_sl = simulate_country(
    df_sl, infectious_period_sl, N_population_sl, beta_estimate_sl, sigma_estimate_sl
)
g_dates, g_I, beta_g, nu_g, N_g = simulate_country(
    df_g, infectious_period_g, N_population_g, beta_estimate_g, sigma_estimate_g
)

R0_l = beta_l / nu_l
R0_sl = beta_sl / nu_sl
R0_g = beta_g / nu_g

plt.figure(figsize=(14, 7))

plt.plot(liberia_dates, liberia_I / 1000, linewidth=2.5, label=f"Liberia (R₀={R0_l:.2f})", color='tab:blue')
plt.plot(sl_dates, sl_I / 1000, linewidth=2.5, label=f"Sierra Leone (R₀={R0_sl:.2f})", color='tab:orange')
plt.plot(g_dates, g_I / 1000, linewidth=2.5, label=f"Guinea (R₀={R0_g:.2f})", color='tab:green')

plt.title("Infected Curve Comparison Across Liberia, Sierra Leone and Guinea\nFor Ebola Outbreak Using SEIR Model", 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Date", fontsize=16, fontweight='bold')
plt.ylabel("Number of Infected Individuals (thousands)", fontsize=16, fontweight='bold')

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.gcf().autofmt_xdate()

# Make tick marks larger
ax.tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)

plt.grid(True, linestyle="--", alpha=0.5, linewidth=1)
plt.legend(loc='best', fontsize=14, framealpha=0.9)
plt.tight_layout()
plt.show()