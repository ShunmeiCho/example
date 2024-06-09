import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import timeit

# Gauss Elimination with Numba
@njit
def gauss_elimination_numba(A, b):
    n = len(b)
    Ab = np.column_stack((A.astype(np.float64), b.astype(np.float64)))

    for i in range(n - 1):  # 遍历每一个主元行
        for j in range(i + 1, n):  # 对于每一个主元行，下方的每一行
            # 计算消去因子 factor
            factor = Ab[j, i] / Ab[i, i]
            # 用主元行的每一个元素乘以消去因子，然后从当前行中减去
            for k in range(i, n + 1):  # 从当前列到最后一列（包括b列）
                Ab[j, k] -= factor * Ab[i, k]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        dot_product = 0.0
        for j in range(i + 1, n):
            dot_product += Ab[i, j] * x[j]
        x[i] = (Ab[i, n] - dot_product) / Ab[i, i]

    return x

# Gauss Elimination without Numba
def gauss_elimination(A, b):
    n = len(b)
    Ab = np.column_stack((A.astype(np.float64), b.astype(np.float64)))

    for i in range(n - 1):  # 遍历每一个主元行
        for j in range(i + 1, n):  # 对于每一个主元行，下方的每一行
            # 计算消去因子 factor
            factor = Ab[j, i] / Ab[i, i]
            # 用主元行的每一个元素乘以消去因子，然后从当前行中减去
            for k in range(i, n + 1):  # 从当前列到最后一列（包括b列）
                Ab[j, k] -= factor * Ab[i, k]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        dot_product = 0.0
        for j in range(i + 1, n):
            dot_product += Ab[i, j] * x[j]
        x[i] = (Ab[i, n] - dot_product) / Ab[i, i]

    return x
# Maximum Relative Error
@njit
def max_relative_error(x_vec, y):
    return np.max(np.abs(x_vec - y)) / np.max(np.abs(y))

# Function to create tri-diagonal matrix
@njit
def create_matrix(n, a, b):
    matrix = np.zeros((n - 1, n - 1))
    np.fill_diagonal(matrix, a)
    np.fill_diagonal(matrix[1:], b)
    np.fill_diagonal(matrix[:, 1:], b)
    return matrix

# Compute Errors and Timings
def compute_errors_and_timings(n_values, num_trials=5):
    average_errors_numba = np.zeros(len(n_values))
    timing_stats_numba = np.zeros((len(n_values), 3))
    
    average_errors_cpu = np.zeros(len(n_values))
    timing_stats_cpu = np.zeros((len(n_values), 3))

    for idx, n in enumerate(n_values):
        errors_numba = np.zeros(num_trials)
        timings_numba = np.zeros(num_trials)
        
        errors_cpu = np.zeros(num_trials)
        timings_cpu = np.zeros(num_trials)
        
        for i in range(num_trials):
            h = (math.pi / 2) / n
            a = 1 - (2 / (h ** 2))
            b = 1 / (h ** 2)
            
            A = create_matrix(n, a, b)
            b_vec = np.zeros(n - 1)
            b_vec[-1] = -b

            # Measure time for Numba version
            start_time = timeit.default_timer()
            x_vec_numba = gauss_elimination_numba(A, b_vec)
            end_time = timeit.default_timer()
            timings_numba[i] = end_time - start_time

            # Measure time for CPU version
            start_time = timeit.default_timer()
            x_vec_cpu = gauss_elimination(A, b_vec)
            end_time = timeit.default_timer()
            timings_cpu[i] = end_time - start_time

            x = np.linspace(h, (n - 1) * h, n - 1)
            sin_x = np.sin(x)

            errors_numba[i] = max_relative_error(x_vec_numba, sin_x)
            errors_cpu[i] = max_relative_error(x_vec_cpu, sin_x)

        average_errors_numba[idx] = np.mean(errors_numba)

        timing_stats_numba[idx, 0] = np.min(timings_numba)
        timing_stats_numba[idx, 1] = np.mean(timings_numba)
        timing_stats_numba[idx, 2] = np.std(timings_numba)
        
        average_errors_cpu[idx] = np.mean(errors_cpu)
        
        timing_stats_cpu[idx, 0] = np.min(timings_cpu)
        timing_stats_cpu[idx, 1] = np.mean(timings_cpu)
        timing_stats_cpu[idx, 2] = np.std(timings_cpu)
        
        print(f'n={n}, Numba Average Error={average_errors_numba[idx]:.2e}, CPU Average Error={average_errors_cpu[idx]:.2e}')
        print(f'Numba Min Time={timing_stats_numba[idx, 0]:.2e}, Numba Avg Time={timing_stats_numba[idx, 1]:.2e}, Numba Std Time={timing_stats_numba[idx, 2]:.2e}')
        print(f'CPU Min Time={timing_stats_cpu[idx, 0]:.2e}, CPU Avg Time={timing_stats_cpu[idx, 1]:.2e}, CPU Std Time={timing_stats_cpu[idx, 2]:.2e}')
    
    return average_errors_numba, timing_stats_numba, average_errors_cpu, timing_stats_cpu

# Function to compute slopes
def compute_slope(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope

# Main Function
def main(start_n=100, end_n=200, step_n=10, num_trials=5):
    n_values = np.arange(start_n, end_n + 1, step_n)
    avg_errors_numba, time_stats_numba, avg_errors_cpu, time_stats_cpu = compute_errors_and_timings(n_values, num_trials)

    slope_error_numba = compute_slope(n_values, avg_errors_numba)
    slope_error_cpu = compute_slope(n_values, avg_errors_cpu)
    slope_time_numba = compute_slope(n_values[1:], time_stats_numba[1:, 1])
    slope_time_cpu = compute_slope(n_values[1:], time_stats_cpu[1:, 1])

    # Plotting Average Error
    plt.figure()

    # plt.plot(n_values, avg_errors_numba, marker='o', label='Numba')
    # plt.plot(n_values, avg_errors_cpu, marker='x', label='CPU')

    plt.plot(n_values, avg_errors_numba, marker='o', label=f'Numba (slope={slope_error_numba:.2f})')
    # plt.plot(n_values, avg_errors_cpu, marker='x', label=f'CPU (slope={slope_error_cpu:.2f})')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Log of Average Error')
    plt.title('Log of Average Error in Gauss Elimination Method')
    plt.legend()
    plt.savefig('/workspace/images/error.png')

    # Plotting Execution Time
    plt.figure(figsize=(10, 6))
    
    # plt.errorbar(n_values[1:], time_stats_numba[1:, 1], yerr=time_stats_numba[1:, 2], label='Numba Avg Time', marker='o', fmt='-', capsize=5)
    # plt.errorbar(n_values[1:], time_stats_cpu[1:, 1], yerr=time_stats_cpu[1:, 2], label='CPU Avg Time', marker='x', fmt='-', capsize=5)
    
    plt.errorbar(n_values[1:], time_stats_numba[1:, 1], yerr=time_stats_numba[1:, 2], label=f'Numba Avg Time (slope={slope_time_numba:.2f})', marker='o', fmt='-', capsize=5)
    plt.errorbar(n_values[1:], time_stats_cpu[1:, 1], yerr=time_stats_cpu[1:, 2], label=f'CPU Avg Time (slope={slope_time_cpu:.2f})', marker='x', fmt='-', capsize=5)
    
    plt.plot(n_values[1:], time_stats_numba[1:, 0], label='Numba Min Time', marker='o', linestyle='--')
    plt.plot(n_values[1:], time_stats_cpu[1:, 0], label='CPU Min Time', marker='x', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time for Gauss Elimination Method')
    plt.legend()
    plt.savefig('/workspace/images/time.png')

if __name__ == '__main__':
    main(start_n=100, end_n=500, step_n=100, num_trials=10)
