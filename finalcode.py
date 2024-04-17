import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Load data
df = pd.read_csv('Vang_Fault.csv')
df = df.apply(pd.to_numeric, errors='coerce')

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

plt.title('Frequencies for IEEE 12-bus System')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

df.drop(columns={'Time'}, inplace=True)

# Function to add AWGN to a given signal
def add_awgn(signal, target_snr_db):
    sig_avg_watts = np.mean(signal ** 2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
    noisy_signal = signal + noise
    return noisy_signal

# Add AWGN to each column of your DataFrame
target_snr_db = 42.85  # Desired target SNR in dB
rows_to_add_noise = range(115, 141)
for column in df.columns:
    df.loc[rows_to_add_noise, column] = add_awgn(df.loc[rows_to_add_noise, column], target_snr_db)

# Output function for rho
# Dimensionality reduction function
ne = 12

def output_rho(Y, e=3.29e-4):
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    error = ne * e
    mode = 0
    for i in range(len(S)):
        if S[i] > error * math.sqrt(Y.size):
            mode = max((i + 1), mode)
        else:
            break
    return mode

Y_final = pd.DataFrame()

# Compression function
def compress(Y, r):
    global Y_final
    U, S, VT = np.linalg.svd(Y, full_matrices=False)
    S = np.diag(S)
    Y_approx = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    Y_final = pd.concat([Y_final, pd.DataFrame(Y_approx)], ignore_index=True)
    return Y_approx

# Compression Ratio function
def CR(Y, rho):
    h = Y.shape[0]
    n = Y.shape[1]
    CR = (h * n) / (rho * (h + n + 1))
    return CR

# Root Mean Squared Error function
def RMSE(Y, Y_approx):
    h = Y.shape[0]
    n = Y.shape[1]
    frobenius_norm = np.linalg.norm(Y - Y_approx, 'fro')
    ans = frobenius_norm / math.sqrt(h * n)
    return ans

# Progressive Partitioning Algorithm
fs = 60
n = 14
l = fs
h = 100
rho_n = 1
phi = 0
n_phi = 0
phi = 0
rho_max = 0
alpha = 0.4
rho_n = 0
sigma = 0
ctr = 0
rho = [1] * 1200  # Increase the length of rho to accommodate the maximum value of time_stamp
Y = pd.DataFrame()  # Initialize a new buffer

def progressive_partitioning(time_stamp, new_buffer=True):
    global phi, rho_max, alpha, rho_n, sigma, ctr, rho, Y, n_phi, Y_approx
    
    if time_stamp >= len(rho):
        print("over")
        return
    
    if new_buffer:
        Y = pd.DataFrame()  # Initialize a new buffer
        Y = pd.concat([Y, df.iloc[time_stamp:time_stamp + 1]], ignore_index=True)
        rho[time_stamp] = output_rho(Y)
    else:
        Y = pd.concat([Y, df.iloc[time_stamp:time_stamp + 1]], ignore_index=True)
        rho[time_stamp] = output_rho(Y)
    
    if phi == 0:
        rho[time_stamp] = output_rho(Y)
        if rho[time_stamp] > rho[time_stamp - 1]:
            phi = 1
            n_phi = 0
            sigma += rho[time_stamp]
            ctr += 1
            rho_max = max(rho_max, rho[time_stamp])
            rho_avg = sigma / ctr
            Y_approx = compress(Y, rho[time_stamp])
            print("The compression ratio is ")
            print(CR(Y, rho[time_stamp]))
            print("RMSE error")
            print(RMSE(Y, Y_approx))
            progressive_partitioning(time_stamp + 1, True)
        elif rho[time_stamp] <= rho[time_stamp - 1]:
            if len(Y) >= h:
                Y_approx = compress(Y, rho[time_stamp])
                print("The compression ratio is ")
                print(CR(Y, rho[time_stamp]))
                print("RMSE error")
                print(RMSE(Y, Y_approx))
                progressive_partitioning(time_stamp + 1, True)
            elif len(Y) < h:
                progressive_partitioning(time_stamp + 1, False)
    elif phi == 1:
        sigma += rho[time_stamp]
        ctr += 1
        rho_max = max(rho_max, rho[time_stamp])
        rho_avg = sigma / ctr
        if rho_avg < alpha * rho_max:
            phi = 2
            Y_approx = compress(Y, rho[time_stamp])
            print("The compression ratio is ")
            print(CR(Y, rho[time_stamp]))
            print("RMSE error")
            print(RMSE(Y, Y_approx))
            progressive_partitioning(time_stamp + 1, True)
        elif rho_avg >= alpha * rho_max:
            if rho[time_stamp] == rho_n:
                n_phi += 1
                if n_phi >= l:
                    phi = 0
                    n_phi = 0
                    Y_approx = compress(Y, rho[time_stamp])
                    print("The compression ratio is ")
                    print(CR(Y, rho[time_stamp]))
                    print("RMSE error")
                    print(RMSE(Y, Y_approx))
                    progressive_partitioning(time_stamp + 1, True)
                else:
                    if len(Y) >= h:
                        Y_approx = compress(Y, rho[time_stamp])
                        print("The compression ratio is ")
                        print(CR(Y, rho[time_stamp]))
                        print("RMSE error")
                        print(RMSE(Y, Y_approx))
                        progressive_partitioning(time_stamp + 1, True)
                    elif len(Y) < h:
                        progressive_partitioning(time_stamp + 1, False)
            else:
                n_phi = 0
                if len(Y) >= h:
                    Y_approx = compress(Y, rho[time_stamp])
                    print("The compression ratio is ")
                    print(CR(Y, rho[time_stamp]))
                    print("RMSE error")
                    print(RMSE(Y, Y_approx))
                    progressive_partitioning(time_stamp + 1, True)
                elif len(Y) < h:
                    progressive_partitioning(time_stamp + 1, False) 

progressive_partitioning(1)

# Plot Estimated Rank vs. Timestamp
plt.figure()
timestamps = list(range(len(rho)))
plt.plot(timestamps, rho, label='Estimated Rank (rho[t])')
plt.title('Estimated Rank vs. Timestamp')
plt.xlabel('Timestamp (t)')
plt.ylabel('Estimated Rank (rho[t])')
plt.legend()
plt.grid(True)
plt.show()

# Plot Columns of Y_final
num_rows = len(Y_final)
total_time = 20  # Total time in seconds
step = total_time / num_rows
x_values = np.arange(0, total_time, step)

for column in Y_final.columns:
    plt.plot(x_values, Y_final[column], label=column)

plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Plot of Columns of Y_final')
plt.legend()
plt.show()

# Comparison of First Column between df and Y_final
df_columns = df.columns
Y_final.columns = df_columns

plt.figure()
plt.plot(df.index, df[df_columns[3]], label=f'Column {df_columns[3]} (df)')
plt.plot(Y_final.index, Y_final[df_columns[3]], label=f'Column {df_columns[3]} (Y_final)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title(f'Comparison of First Column ({df_columns[3]}) between df and Y_final')
plt.legend()
plt.grid(True)
plt.show()
