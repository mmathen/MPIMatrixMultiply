# MPI Matrix Multiplication Benchmark

This project implements and benchmarks matrix multiplication using MPI for parallel processing. It includes tools for running benchmarks and visualizing performance results.

## Features

- Distributed matrix multiplication using MPI
- Performance benchmarking tool
- Visualization of performance metrics
- Scalability analysis

## Requirements

- Python 3.x
- mpi4py
- numpy
- matplotlib
- pandas
- seaborn

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Benchmark

Run the benchmark using:
```bash
python run_benchmark.py
```

This will run benchmarks with different matrix sizes and process counts.

### Running the Visualization

Generate performance plots using:
```bash
python visualization.py
```

### Running a Single Test

Run a single matrix multiplication test using MPI:
```bash
mpiexec -n <number_of_processes> python matrix_multiply.py
```

Example:
```bash
mpiexec -n 4 python matrix_multiply.py
```
Run on AWS multi cluster matrix multiplication test using MPI:
```bash
mpiexec -np <number_of_processes> --hostfile ~/my_mpi_hosts.txt /usr/bin/python3 matrix_multiply.py <matrix_size>
```

Example:
```bash
mpiexec -np 4 --hostfile ~/my_mpi_hosts.txt  /usr/bin/python3 matrix_multiply.py 2000
```

## Performance Metrics

The program measures and visualizes:
- Sequential execution time
- Distributed execution time
- Speedup factor (sequential time / distributed time)
- Efficiency
- Strong scaling performance
- Execution time vs matrix size
