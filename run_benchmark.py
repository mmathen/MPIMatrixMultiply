import subprocess
import os
from time import sleep

def run_benchmark(matrix_sizes, process_counts, versions):
    """
    Run benchmarks for different matrix sizes, process counts, and versions.
    
    Args:
        matrix_sizes: List of matrix sizes to test
        process_counts: List of process counts to test
        versions: List of version names to test (e.g., ['matrix_multiply.py', 'optimized_matrix_multiply.py'])
    """
    # Clear existing results files before starting
    for version in versions:
        # Construct the expected results filename for each version
        base_name = os.path.splitext(version)[0]
        filename = f"{base_name}_performance_results.csv"
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Cleared existing results file: {filename}")
            except OSError as e:
                print(f"Error clearing {filename}: {e}")
    
    print("Starting benchmarks...")
    
    for version in versions:
        print(f"\n--- Testing version: {version} ---")
        for n in matrix_sizes:
            print(f"\n--- Testing matrix size: {n}x{n} ---")
            for p in process_counts:
                print(f"Running with {p} processes...")
                
                # Command to run the MPI program, passing matrix size as argument
                cmd = ["mpiexec", "-n", str(p), "python", version, str(n)]
                print(f"Running command: {' '.join(cmd)}")
                
                try:
                    # Use subprocess.run with timeout to prevent infinite hangs
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300) # 5-minute timeout
                    
                    if result.returncode != 0:
                        print(f"Error or non-zero exit code for n={n}, p={p} in version {version}")
                        print(f"STDOUT:\n{result.stdout}")
                        print(f"STDERR:\n{result.stderr}")
                        # Optionally, you might want to log this failure
                        continue # Move to the next benchmark if one fails
                    else:
                        print(f"Benchmark for n={n}, p={p} completed successfully.")
                        print(f"STDOUT:\n{result.stdout}") # Print stdout for successful runs too
                        # Check stderr for any warnings/errors even on success
                        if result.stderr:
                            print(f"STDERR (warnings/errors):\n{result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"TIMEOUT: Benchmark for n={n}, p={p} in version {version} timed out after 300 seconds.")
                    # Handle the timeout, perhaps kill the process group if it's still running
                    # (subprocess.run with timeout already handles termination)
                except Exception as e:
                    print(f"An unexpected error occurred for n={n}, p={p} in version {version}: {e}")
                    
                # A small delay between runs might still be beneficial for system stability
                sleep(1) 

    print("\nBenchmarking complete! Results should be saved to:")
    for version in versions:
        base_name = os.path.splitext(version)[0]
        print(f"- {base_name}_performance_results.csv")

def main():
    process_counts = [1, 2,4,8] # Start with 1 and 2 to easily test
    # matrix_sizes = [1000] # For initial quick test
    matrix_sizes = [500, 1000, 1001, 2000, 4000] # Test with N not divisible by process_count
    
    versions = ['matrix_multiply.py'] 
    
    run_benchmark(matrix_sizes, process_counts, versions)

if __name__ == "__main__":
    main()