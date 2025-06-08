import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results(version_name):
    """
    Load performance results for a specific version.
    """
    filename = f"{version_name}_performance_results.csv"
    try:
        if not Path(filename).exists():
            print(f"Warning: {filename} not found. Skipping this version.")
            return None
        
        # Read the file with error handling
        df = pd.read_csv(filename)
        print(f"Successfully loaded {filename}")
        return df
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        return None

def plot_speedup_vs_processes(df, version_name):
    """
    Plot speedup vs number of processes for different matrix sizes.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='processes', y='speedup', hue='matrix_size', marker='o')
    plt.axhline(y=1, color='r', linestyle='--')  # Ideal speedup line
    plt.title(f'Speedup vs Number of Processes - {version_name}')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.legend(title='Matrix Size')
    plt.savefig(f'speedup_vs_processes_{version_name}.png')
    plt.close()

def plot_efficiency_vs_processes(df, version_name):
    """
    Plot efficiency vs number of processes for different matrix sizes.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='processes', y='efficiency', hue='matrix_size', marker='o')
    plt.axhline(y=1, color='r', linestyle='--')  # Ideal efficiency line
    plt.title(f'Efficiency vs Number of Processes - {version_name}')
    plt.xlabel('Number of Processes')
    plt.ylabel('Efficiency')
    plt.grid(True)
    plt.legend(title='Matrix Size')
    plt.savefig(f'efficiency_vs_processes_{version_name}.png')
    plt.close()

def plot_execution_time_vs_matrix_size(df, version_name):
    """
    Plot execution time vs matrix size for different process counts.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot sequential times
    seq_df = df[df['processes'] == 1]
    plt.plot(seq_df['matrix_size'], seq_df['sequential_time'], 'r-', label='Sequential')
    
    # Plot parallel times for different process counts
    for p in df['processes'].unique():
        if p > 1:
            p_df = df[df['processes'] == p]
            plt.plot(p_df['matrix_size'], p_df['distributed_time'], label=f'Parallel (p={p})')
    
    plt.title(f'Execution Time vs Matrix Size - {version_name}')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig(f'execution_time_vs_matrix_size_{version_name}.png')
    plt.close()

def plot_strong_scaling(df, version_name):
    """
    Plot strong scaling analysis: fixed problem size, varying number of processes.
    """
    plt.figure(figsize=(12, 6))
    
    # Get the largest matrix size tested
    max_size = df['matrix_size'].max()
    
    # Filter data for the largest matrix size
    strong_df = df[df['matrix_size'] == max_size]
    
    # Calculate theoretical speedup
    strong_df['theoretical_speedup'] = strong_df['processes']
    
    plt.plot(strong_df['processes'], strong_df['theoretical_speedup'], 'r--', label='Theoretical Speedup')
    plt.plot(strong_df['processes'], strong_df['speedup'], 'b-', marker='o', label='Actual Speedup')
    
    plt.title(f'Strong Scaling Analysis - {version_name} (Matrix Size: {max_size}x{max_size})')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'strong_scaling_{version_name}.png')
    plt.close()

def main():
    # Define versions to visualize
    versions = ['matrix_multiply']#, 'optimized_matrix_multiply.py']
    
    successful_versions = []
    
    for version in versions:
        print(f"\nGenerating plots for {version}")
        df = load_results(version)
        if df is None:
            continue
        
        # Create plots for this version
        plot_speedup_vs_processes(df, version)
        plot_efficiency_vs_processes(df, version)
        plot_execution_time_vs_matrix_size(df, version)
        plot_strong_scaling(df, version)
        
        successful_versions.append(version)
    
    print("\nVisualization complete! Generated plots:")
    if not successful_versions:
        print("No valid data files found to generate plots.")
    else:
        for version in successful_versions:
            print(f"For {version}:")
            print(f"- speedup_vs_processes_{version}.png")
            print(f"- efficiency_vs_processes_{version}.png")
            print(f"- execution_time_vs_matrix_size_{version}.png")
            print(f"- strong_scaling_{version}.png")
            print()

if __name__ == "__main__":
    main()
