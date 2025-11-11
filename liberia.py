import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Repeat process for each country
infectious_period = 11  # Use average infectious period to approximate https://www.who.int/news-room/fact-sheets/detail/ebola-disease
N_population = 7040000 # population of liberia

df = pd.read_csv("l.csv")
df['new_cases'] = df['cum_cases'].diff().fillna(0)
df['new_deaths'] = df['cum_deaths'].diff().fillna(0)
df['active_cases'] = df['cum_cases'] - df['cum_cases'].shift(infectious_period).fillna(0)
df['recovered'] = df['cum_cases'].shift(infectious_period).fillna(0) - df['cum_deaths']
df['removed'] = df['recovered'] + df['cum_deaths']

#I Value
df['I'] = df['active_cases']

#R Value
df['R'] = df['removed']

#Susceptible
df['S'] = N_population - df['I'] - df['R']

# Normalise to allow for comparisons between different countries
df['s'] = df['S'] 
df['i'] = df['I'] 
df['r'] = df['R'] 

# Average infectious period
nu = 1 / infectious_period
df['di_dt'] = df['i'].diff().fillna(0)

#Estimate of beta based on the new_cases
beta_estimate = 1.547

#Calculate R0, initial value 
R0 = beta_estimate / nu
S, I, R = [N_population-1], [1], [0]
dt = 1  # 1 day

for t in range(1, len(df)):
    S_next = S[-1] - beta_estimate * S[-1] * I[-1] / N_population * dt
    I_next = I[-1] + (beta_estimate * S[-1] * I[-1] / N_population - nu * I[-1]) * dt
    R_next = R[-1] + nu * I[-1] * dt
    S.append(S_next)
    I.append(I_next)
    R.append(R_next)

df['S_model'], df['I_model'], df['R_model'] = S, I, R
x = np.array(df['Date'])
y = np.array(df['S_model'], dtype=float)
plt.figure(figsize=(100, 10))
plt.scatter(x, y, c='blue', marker='o', edgecolors='black', alpha=0.7)

plt.title("Sierra Leone I(t) Plot")
plt.xlabel("Date")
plt.ylabel("I(t) Value From Euler's Method")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Repeat process for each country
infectious_period = 11  # Use average infectious period to approximate https://www.who.int/news-room/fact-sheets/detail/ebola-disease
N_population = 4660000 # population of Sierra Leon

df = pd.read_csv("l.csv")
df['new_cases'] = df['cum_cases'].diff().fillna(0)
df['new_deaths'] = df['cum_deaths'].diff().fillna(0)
df['active_cases'] = df['cum_cases'] - df['cum_cases'].shift(infectious_period).fillna(0)
df['recovered'] = df['cum_cases'].shift(infectious_period).fillna(0) - df['cum_deaths']
df['removed'] = df['recovered'] + df['cum_deaths']

#I Value
df['I'] = df['active_cases']

#R Value
df['R'] = df['removed']

#Susceptible
df['S'] = N_population - df['I'] - df['R']

# Normalise to allow for comparisons between different countries
df['s'] = df['S'] 
df['i'] = df['I'] 
df['r'] = df['R'] 

# Average infectious period
nu = 1 / infectious_period
df['di_dt'] = df['i'].diff().fillna(0)

#Estimate of beta based on the new_cases
beta_estimate = 1.365

#Calculate R0, initial value 
R0 = beta_estimate / nu
S, I, R = [N_population-1], [1], [0]
dt = 1  # 1 day

for t in range(1, len(df)):
    S_next = S[-1] - beta_estimate * S[-1] * I[-1] / N_population * dt
    I_next = I[-1] + (beta_estimate * S[-1] * I[-1] / N_population - nu * I[-1]) * dt
    R_next = R[-1] + nu * I[-1] * dt
    S.append(S_next)
    I.append(I_next)
    R.append(R_next)

df['S_model'], df['I_model'], df['R_model'] = S, I, R

#Plot for each model
x = np.array(df['Date'])
y = np.array(df['S_model'], dtype=float)
plt.figure(figsize=(100, 10))
plt.scatter(x, y, c='blue', marker='o', edgecolors='black', alpha=0.7)

plt.title("Sierra Leone I(t) Plot")
plt.xlabel("Date")
plt.ylabel("I(t) Value From Euler's Method")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

