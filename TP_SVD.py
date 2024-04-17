import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import sklearn.datasets
import sklearn.preprocessing
import math

df = pd.read_csv('Ang_Freq_Fault.csv')

df = df.apply(pd.to_numeric, errors='coerce')

import pandas as pd
import matplotlib.pyplot as plt


# Define time parameters
total_samples = len(df)  # Total samples per bus
total_time = 20  # Total time in seconds
sampling_rate = 60  # Samples per second

# Create time array
time = np.linspace(0, total_time, total_samples)

# Plotting
plt.figure(figsize=(12, 6))

for bus in df.columns:
    if bus == 'Time':
        continue
    plt.plot(time, df[bus], label=bus)

# Highlight disturbance at t=9 sec
disturbance_time = 1.8  # Time of disturbance in seconds
plt.axvline(x=disturbance_time, color='red', linestyle='--', label='Disturbance')

plt.title('Frequencies for IEEE 14-bus System')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

df.drop(columns={'Time'}, inplace=True)

import numpy as np
import pandas as pd

# Function to add AWGN to a given signal
def add_awgn(signal, target_snr_db):
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(signal ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    
    # Calculate noise and convert it to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    # Generate samples of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
    
    # Add noise to original signal
    noisy_signal = signal + noise
    
    return noisy_signal


# Add AWGN to each column of your DataFrame
target_snr_db = 50 # Desired target SNR in dB
rows_to_add_noise = range(115, 141)
for column in df.columns:
    df.loc[rows_to_add_noise, column] = add_awgn(df.loc[rows_to_add_noise, column], target_snr_db)

# Now your DataFrame 'data' contains AWGN added to each column
print(df.head())

ne = 14 # this is dimensionality reduction function
def output_rho(Y, e=3.29e-4):
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    error = ne*e;
    mode = 0
    for i in range(len(S)):
        if(S[i] > error * math.sqrt(Y.size)):
            mode = max((i+1), mode)
        else:
            break
    return mode

import pandas as pd
import numpy as np
import math

Y_final = pd.DataFrame()

def compress(Y, r):
    global Y_final
    U, S, VT = np.linalg.svd(Y, full_matrices=False)
    S = np.diag(S)
    Y_approx = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    Y_final = pd.concat([Y_final, pd.DataFrame(Y_approx)], ignore_index=True)
    return Y_approx;

def CR(Y, rho):
    h = Y.shape[0]
    n = Y.shape[1]
    CR = (h * n) / (rho * (h + n + 1))
    return CR

def RMSE(Y, Y_approx):
    h = Y.shape[0]
    n = Y.shape[1]
    frobenius_norm = np.linalg.norm(Y - Y_approx, 'fro')
    ans = frobenius_norm / math.sqrt(h * n)
    return ans

# The rest of your code...

import pandas as pd

fs = 60
n = 14
l = fs
h = 100
rho_n = 1
phi = 0
n_phi = 0
# Global variables
phi = 0
rho_max = 0
alpha = 0.4
rho_n = 0
sigma = 0
ctr = 0
rho = [1] * 1200  # Increase the length of rho to accommodate the maximum value of time_stamp
Y = pd.DataFrame()  # Initialize a new buffer


def progressive_partitioning(time_stamp, new_buffer=True):
    global phi, rho_max, alpha, rho_n, sigma, ctr, rho, Y, n_phi, Y_approx  # Declare n_phi as a global variable
    
    if time_stamp >= len(rho):  # Check if time_stamp exceeds the length of rho
        print("over")
        return
    
    if new_buffer:
        Y = pd.DataFrame()  # Initialize a new buffer
        #print("Initializing new buffer......")
        Y = pd.concat([Y, df.iloc[time_stamp:time_stamp+1]], ignore_index=True)
        rho[time_stamp] = output_rho(Y)
    else:
        #print("Buffer not yet terminated!")
        Y = pd.concat([Y, df.iloc[time_stamp:time_stamp+1]], ignore_index=True)
        rho[time_stamp] = output_rho(Y)
    
    if phi == 0:
        #print("I am inside phi = 0 " + str(time_stamp))
        rho[time_stamp] = output_rho(Y)
        if rho[time_stamp] > rho[time_stamp - 1]:
            phi = 1
            n_phi = 0
            sigma += rho[time_stamp]
            ctr += 1
            rho_max = max(rho_max, rho[time_stamp])
            rho_avg = sigma / ctr
            #print("now phi will be 1!!!")
            Y_approx = compress(Y, rho[time_stamp])
            print("The compression ratio is ")
            print(CR(Y, rho[time_stamp]))
            print("RMSE error")
            print(RMSE(Y, Y_approx))
            progressive_partitioning(time_stamp+1, True)  # starts new buffer
        elif rho[time_stamp] <= rho[time_stamp - 1]:
            if len(Y) >= h:
                #print("left side me hu abhi")
                Y_approx = compress(Y, rho[time_stamp])
                print("The compression ratio is ")
                print(CR(Y, rho[time_stamp]))
                print("RMSE error")
                print(RMSE(Y, Y_approx))
                progressive_partitioning(time_stamp+1, True)  # starts new buffer
            elif len(Y) < h:
                progressive_partitioning(time_stamp+1, False)
    elif phi == 1:
        #print("Inside phi = 1")
        sigma += rho[time_stamp]
        ctr += 1
        rho_max = max(rho_max, rho[time_stamp])
        rho_avg = sigma / ctr
        if rho_avg < alpha * rho_max:
            #print("phi 2 hu abhi")
            phi = 2
            #print("right me hu abhi")
            Y_approx = compress(Y, rho[time_stamp])
            print("The compression ratio is ")
            print(CR(Y, rho[time_stamp]))
            print("RMSE error")
            print(RMSE(Y, Y_approx))
            progressive_partitioning(time_stamp+1, True)  # starts new buffer
        elif rho_avg >= alpha * rho_max:
            if rho[time_stamp] == rho_n:
                n_phi += 1
                if n_phi >= l:
                    phi = 0
                    n_phi = 0
                    #print("Right beech me hu")
                    Y_approx = compress(Y, rho[time_stamp])
                    print("The compression ratio is ")
                    print(CR(Y, rho[time_stamp]))
                    print("RMSE error")
                    print(RMSE(Y, Y_approx))
                    progressive_partitioning(time_stamp+1, True)  # starts new buffer
                else:
                    if len(Y) >= h:
                        #print("left me hu bhai")
                        Y_approx = compress(Y, rho[time_stamp])
                        print("The compression ratio is ")
                        print(CR(Y, rho[time_stamp]))
                        print("RMSE error")
                        print(RMSE(Y, Y_approx))
                        progressive_partitioning(time_stamp+1, True)  # starts new buffer
                    elif len(Y) < h:
                        progressive_partitioning(time_stamp+1, False)
            else:
                n_phi = 0
                if len(Y) >= h:
                    #print("left me hu bhai")
                    Y_approx = compress(Y, rho[time_stamp])
                    print("The compression ratio is ")
                    print(CR(Y, rho[time_stamp]))
                    print("RMSE error")
                    print(RMSE(Y, Y_approx))
                    progressive_partitioning(time_stamp+1, True)  # starts new buffer
                elif len(Y) < h:
                    progressive_partitioning(time_stamp+1, False) 
progressive_partitioning(1)

import plotly.graph_objects as go

# Convert range object to list for the x-axis values
timestamps = list(range(len(rho)))

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=timestamps, y=rho, mode='lines', name='Estimated Rank (rho[t])'))
fig.update_layout(title='Estimated Rank vs. Timestamp',
                  xaxis_title='Timestamp (t)',
                  yaxis_title='Estimated Rank (rho[t])',
                  template='plotly_white')
fig.show()


import numpy as np

# Assuming Y_final is your DataFrame
num_rows = len(Y_final)
total_time = 20  # Total time in seconds

# Calculate time interval between each point
step = total_time / num_rows

# Generate x-axis values
x_values = np.arange(0, total_time, step)

import matplotlib.pyplot as plt

# Plot each column
for column in Y_final.columns:
    plt.plot(x_values, Y_final[column], label=column)

# Add labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Plot of Columns of Y_final')
plt.legend()

# Show plot
plt.show()

import plotly.graph_objects as go

# Assuming df is your DataFrame with column names
df_columns = df.columns

# Rename the columns of Y_final
Y_final.columns = df_columns

# Create a figure
fig = go.Figure()

# Plot the first column of df
fig.add_trace(go.Scatter(x=df.index, y=df[df_columns[3]], mode='lines', name=f'Column {df_columns[3]} (df)'))

# Plot the first column of Y_final
fig.add_trace(go.Scatter(x=Y_final.index, y=Y_final[df_columns[3]], mode='lines', name=f'Column {df_columns[3]} (Y_final)'))

# Add axis labels and title
fig.update_layout(
    xaxis_title='Index',
    yaxis_title='Value',
    title=f'Comparison of First Column ({df_columns[3]}) between df and Y_final'
)

# Show the plot
fig.show()


#print(RMSE(df, Y_final))



