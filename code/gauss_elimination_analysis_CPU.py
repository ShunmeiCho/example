import math
import numpy as np
import matplotlib.pyplot as plt
import timeit


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

# 最大相对误差

def max_relative_error(x_vec, y):
    return np.max(np.abs(x_vec - y)) / np.max(np.abs(y))

# 上下三角行列‘0’を作成する関数
def create_matrix(n, a, b):
    matrix = np.zeros((n - 1, n - 1))
    np.fill_diagonal(matrix, a)
    np.fill_diagonal(matrix[1:], b)
    np.fill_diagonal(matrix[:, 1:], b)
    return matrix

# Error and time 
def compute_errors_and_timings(n_values, num_trials=5):
    average_errors = np.zeros(len(n_values))
    timing_stats = np.zeros((len(n_values), 3))

    for idx, n in enumerate(n_values):
        errors = np.zeros(num_trials)
        timings = np.zeros(num_trials)
        
        for i in range(num_trials):
            h = (math.pi / 2) / n
            a = 1 - (2 / (h ** 2))
            b = 1 / (h ** 2)
            
            A = create_matrix(n, a, b)
            b_vec = np.zeros(n - 1)
            b_vec[-1] = -b

            start_time = timeit.default_timer()
            x_vec = gauss_elimination(A, b_vec)
            end_time = timeit.default_timer()

            x = np.linspace(h, (n - 1) * h, n - 1)
            sin_x = np.sin(x)

            errors[i] = max_relative_error(x_vec, sin_x)
            timings[i] = end_time - start_time

        average_errors[idx] = np.mean(errors)
        timing_stats[idx, 0] = np.min(timings)
        timing_stats[idx, 1] = np.mean(timings)
        timing_stats[idx, 2] = np.std(timings)
        
        print(f'n={n}, Average Error={average_errors[idx]:.2e}, Min Time={timing_stats[idx, 0]:.2e}, Avg Time={timing_stats[idx, 1]:.2e}, Std Time={timing_stats[idx, 2]:.2e}')
    
    return average_errors, timing_stats

def main(start_n=100, end_n=200, step_n=10, num_trials=5):
    n_values = np.arange(start_n, end_n + 1, step_n)
    average_errors, timing_stats = compute_errors_and_timings(n_values, num_trials)

    # 绘制平均误差
    plt.figure()
    plt.plot(n_values, average_errors, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Log of Average Error')
    plt.title('Log of Average Error in Gauss Elimination Method')
    plt.savefig('/workspace/images/error.png')

    # 绘制执行时间
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_values[1:], timing_stats[1:, 1], yerr=timing_stats[1:, 2], label='Avg Time', marker='o', fmt='-', capsize=5)
    plt.plot(n_values[1:], timing_stats[1:, 0], label='Min Time', marker='o', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time for Gauss Elimination Method')
    plt.legend()
    plt.savefig('/workspace/images/time.png')

if __name__ == '__main__':
    main()
    # main(start_n=1000, end_n=2000, step_n=100, num_trials=100)
