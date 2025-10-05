# SpMM-data-structures

A Julia-based benchmarking suite for evaluating sparse matrix multiplication performance across varying sparsity levels, with a focus on the 10-50% sparsity range common in modern machine learning applications.

## Overview

The project addresses the performance gap in sparse computing: while current sparse matrix libraries optimize for extremely low sparsity (1-3% nonzeros), many real-world applications exhibit moderate sparsity levels (10-50%) where traditional sparse formats offer little to no performance benefit over dense implementations.

### Benchmarking Existing Libraries

- Compares sparse matrix multiplication performance of existing Julia sparse matrix format libraries (SparseArrays, SuiteSparseGraphBLAS, Finch) against dense implementations
- Evaluates performance across a range of sparsity levels (particularly 1e-6-50% nonzeros)
- Identifies crossover points where sparse implementations become beneficial

### Custom Data Structure Implementation

- Custom COO/ CSC format: Implement sparse matrix multiplication for the following pre-existing formats to compare with formats from existing libraries
- Bitmap/Bytemap-based formats: Implements sparse matrix representations using bitmaps/bytemaps to track nonzero positions