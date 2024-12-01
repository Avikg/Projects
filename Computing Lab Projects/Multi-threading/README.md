# MultiThreading and Synchronization Assignment (CS69011)

## Overview

This repository contains the solution to **Assignment: MultiThreading and Synchronization** as part of the **CS69011 Computing Lab** course. The assignment involves using the **pthread** library to handle multithreading tasks, ensuring thread-safe operations through **mutex locks** and **semaphores**. The task focuses on finding approximate shortest paths in a dynamic social graph.

---

## Files Included

```
.
├── proj3.c                   # Main C file containing the implementation
├── loc-brightkite_edges.txt  # Input file for the graph in edge list format
├── path_found.log            # Log of paths found, removed, or not found
├── update.log                # Log of graph updates (additions/removals)
├── test.py                   # Python script to validate correctness
├── report.pdf                # Report answering assignment questions
```

---

## Requirements

### Software
- **Linux OS**: Required for compiling and running the code.
- **GCC Compiler**: To compile the C program.
- **Python 3**: For validation with the `test.py` script.

### Libraries
- **Pthread Library**: For multithreading.
- Python modules:
  - Standard libraries such as `os`, `sys`, and `logging`.

---

## Compilation and Execution

### Compilation
Compile the C program using GCC with the pthread library:

```bash
gcc proj3.c -o assignment8 -pthread
```

### Execution
Run the compiled program:

```bash
./assignment8
```

### Outputs
- **path_found.log**: Contains paths found, removed, or not found.
- **update.log**: Logs of edge additions and removals.
- **final_graph.edgelist**: (Generated if implemented) Final state of the graph after processing all paths.

---

## Features and Workflow

### Graph Setup
- Reads the graph from `loc-brightkite_edges.txt` in edge list format.
- Creates an adjacency list representation for efficient shortest path calculations.

### Landmark Nodes
- Chooses 50 random nodes and 50 highest-degree nodes as landmarks.
- Partitions nodes into 100 groups, assigning each group to a landmark node.

### Threads
1. **Graph Update Threads**:
   - Randomly add or remove edges.
   - Write operations to `update.log`.

2. **Path Finder Threads**:
   - Compute paths between nodes in partitions and their respective landmarks.
   - Ensure paths remain consistent with graph updates.

3. **Path Stitcher Threads**:
   - Approximate shortest paths between node pairs using landmark nodes.
   - Write results to `path_found.log`.

### Synchronization
- **Mutex Locks and Semaphores**:
  - Prevent race conditions.
  - Ensure proper priority between threads (e.g., `path_stitcher` over `path_finder`).

---

## Validation

Run the provided Python script to validate correctness:

```bash
python3 test.py
```

The script parses logs to:
1. Verify `PATH_NOT_FOUND` entries are valid.
2. Ensure all found paths are consistent with the graph’s state.

---

## Report

Refer to the `report.pdf` for:
1. Data structures used and their memory usage.
2. Locks and semaphores applied, and their effectiveness.
3. Debugging techniques and challenges encountered.

---

## Author

**Avik Pramanick**  
**Roll No:** 23CS60R78  
**Course:** CS69011 Computing Lab  
