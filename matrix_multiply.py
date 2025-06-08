import numpy as np
from mpi4py import MPI
import time
import sys # Import sys for command-line arguments
import csv # Import csv for direct writing of results from rank 0
import os # Import os for file operations

def sequential_matrix_multiply(A, B):
    return np.dot(A, B)

def distributed_matrix_multiply(A, B, comm, rank, size):
    n = A.shape[0]
    
    rows_per_process = [n // size + (1 if i < n % size else 0) for i in range(size)]
    
    # Calculate displacements for Scatterv and Gatherv
    # Note: MPI.DOUBLE is for numpy.float64, adjust if using other dtypes
    displacements_elements_A = [sum(rows_per_process[:i]) * n for i in range(size)]
    sendcounts_elements_A = [count * n for count in rows_per_process] 
    
    local_A = np.zeros((rows_per_process[rank], n), dtype=A.dtype)
    
    if rank == 0:
        # Scatterv expects a tuple (data, sendcounts, displacements, datatype)
        comm.Scatterv([A, sendcounts_elements_A, displacements_elements_A, MPI.DOUBLE], local_A)
    else:
        # Non-root processes receive
        comm.Scatterv(None, local_A) 
    
    # Broadcast matrix B to all processes
    comm.Bcast(B, root=0)
    
    local_C = np.dot(local_A, B)
    
    # Prepare for Gatherv
    recvcounts_elements_C = [count * n for count in rows_per_process]
    
    if rank == 0:
        C = np.zeros((n, n), dtype=A.dtype)
        # Gatherv expects a tuple (data, recvcounts, displacements, datatype) for root
        comm.Gatherv(local_C, [C, recvcounts_elements_C, displacements_elements_A, MPI.DOUBLE], root=0)
        return C
    else:
        # Non-root processes send
        comm.Gatherv(local_C, None, root=0)
        return None

def measure_performance(n): # Pass n as an argument
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Use n passed as argument
    
    if rank == 0:
        print(f"Matrix size: {n}x{n} on {size} processes (Rank {rank})")
        
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Measure sequential performance
        start_time = time.time()
        C_seq = sequential_matrix_multiply(A, B)
        seq_time = time.time() - start_time
        print(f"Sequential execution time: {seq_time:.6f} seconds (Rank {rank})")
        
        # Measure distributed performance
        comm.Barrier() 
        start_time = time.time()
        C_dist = distributed_matrix_multiply(A, B, comm, rank, size)
        comm.Barrier() 
        dist_time = time.time() - start_time
        print(f"Distributed execution time: {dist_time:.6f} seconds (Rank {rank})")
        
        # Verification
        if np.allclose(C_seq, C_dist):
            print("Results match (sequential vs distributed)!")
        else:
            print("WARNING: Results DO NOT match!")
                    # Handle speedup/efficiency calculation for size=1 or very small times
        if size == 1:
            # For a single process, distributed time is conceptually the sequential time,
            # speedup and efficiency are 1.0 (or not applicable)
            speedup = 1.0
            efficiency = 1.0
            print(f"Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}% (Rank {rank})")
        elif dist_time == 0: # Catch cases where distributed time is still zero for other sizes
            print("Warning: Distributed execution time was 0, cannot calculate speedup/efficiency.")
            speedup = np.inf # Or some large value indicating very fast
            efficiency = np.inf
        else:
            speedup = seq_time / dist_time
            efficiency = speedup / size
            print(f"Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}% (Rank {rank})")
        
        # speedup = seq_time / dist_time
        # efficiency = speedup / size
        # print(f"Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}% (Rank {rank})")
        
        results = {
            'matrix_size': n,
            'processes': size,
            'sequential_time': seq_time,
            'distributed_time': dist_time,
            'speedup': speedup,
            'efficiency': efficiency
        }
        return results
    else:
        # Non-root processes
        A = np.empty((n, n), dtype=np.float64) 
        B = np.empty((n, n), dtype=np.float64) # B will be broadcast, so empty is fine
        comm.Barrier() # Ensure all processes are ready before timing starts
        distributed_matrix_multiply(A, B, comm, rank, size)
        comm.Barrier() # Ensure all processes are ready before timing ends
        return None 

def save_results(results, filename): # Pass filename to save to
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['matrix_size', 'processes', 'sequential_time', 'distributed_time', 'speedup', 'efficiency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)

if __name__ == "__main__":          
    if len(sys.argv) < 2:
        print("Usage: mpiexec -n <num_processes> python matrix_multiply.py <matrix_size>")
        sys.exit(1)
    
    try:
        matrix_size = int(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid matrix size '{sys.argv[1]}'. Must be an integer.")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Dynamically determine the filename based on the script name
    script_name = os.path.basename(sys.argv[0])
    base_name = os.path.splitext(script_name)[0]
    filename = f"{base_name}_performance_results.csv"

    results = measure_performance(matrix_size)
    
    if rank == 0 and results is not None:
        save_results(results, filename)