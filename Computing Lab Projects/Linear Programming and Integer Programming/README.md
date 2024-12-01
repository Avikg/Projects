# Linear Programming Assignment (CS69011)

## Overview

This repository contains solutions for **Assignment 5: Linear Programming** from the **CS69011 Computing Lab** course. The assignment involves solving linear programming problems using Python and LP solvers.

### Assignment Breakdown:
- **Part A**: Focused on production planning optimization.
- **Part B**: Focused on sweet box optimization problems.

## Structure

```
.
├── PartA/
│   ├── 23CS60R78_Q1.py        # Solution script for Question 1
│   ├── 23CS60R78_Q2.py        # Solution script for Question 2
│   ├── Summary_Q1.txt         # Output summary for Question 1
│   ├── Summary_Q2.txt         # Output summary for Question 2
│   ├── input_Q1.txt           # Input file for Question 1
│   ├── input_Q2.txt           # Input file for Question 2
├── PartB/
│   ├── 23CS60R78_Q3.py        # Solution script for Question 3
│   ├── 23CS60R78_Q4.py        # Solution script for Question 4
│   ├── Summary_Q3.txt         # Output summary for Question 3
│   ├── Summary_Q4.txt         # Output summary for Question 4
│   ├── input_Q3.txt           # Input file for Question 3
│   ├── input_Q4.txt           # Input file for Question 4
└── README.md                  # This file
```

---

## Questions

### Part A

#### Question 1: Basic Production Planning
- **Objective**: Maximize profit by determining the optimal production plan for `N` products using `M` resources.
- **Input**:
  - Number of products (`N`) and resources (`M`).
  - Profit per unit for each product.
  - Availability of resources.
  - Resource consumption matrix for each product.
- **Output**:
  - Optimal production quantities for each product.
  - Maximum achievable profit.

#### Question 2: Production Planning with Production Capacity Constraints
- **Objective**: Maximize profit under additional constraints on the maximum production capacity of each product.
- **Input**: Same as Q1, with an added constraint for maximum production capacity.
- **Output**:
  - Optimal production quantities for each product considering capacity constraints.
  - Maximum achievable profit.

---

### Part B

#### Question 3: Sweet Box Problem
- **Objective**: Maximize the price of a sweet box while ensuring each sweet fits in the box with no overlap.
- **Input**:
  - Number of sweet types (`k`).
  - Dimensions of the box (`m`, `n`).
  - Dimensions (`x`, `y`) and market prices of each sweet.
- **Output**:
  - Size of each sweet used in the box.
  - Maximum cost of the box.

#### Question 4: Relaxed Sweet Box Problem
- **Objective**: Maximize the price of a sweet box, allowing fractional sweets with cost proportional to size.
- **Input**: Same as Q3.
- **Output**:
  - Size of each sweet used in the box (including fractional parts).
  - Maximum cost of the box.

---

## Requirements

- **Python**: 3.8 or higher.
- Libraries:
  - `ortools`
  - `scipy`

Install required libraries using:

```bash
pip install ortools scipy
```

---

## Usage

### Running the Scripts

1. Prepare the input files (`input_Q1.txt`, `input_Q2.txt`, `input_Q3.txt`, `input_Q4.txt`) following the format described in the assignment.
2. Navigate to the respective part directory and run the scripts:

   **Part A:**
   ```bash
   python PartA/23CS60R78_Q1.py PartA/input_Q1.txt
   python PartA/23CS60R78_Q2.py PartA/input_Q2.txt
   ```

   **Part B:**
   ```bash
   python PartB/23CS60R78_Q3.py PartB/input_Q3.txt
   python PartB/23CS60R78_Q4.py PartB/input_Q4.txt
   ```

3. Outputs will be saved as `Summary_Q1.txt`, `Summary_Q2.txt`, `Summary_Q3.txt`, and `Summary_Q4.txt` in their respective directories.

---

## Notes

- Modify the scripts as needed but ensure only LP solvers are used.
- Follow proper input and output formats as described in the assignment instructions.

---

## Author

**Avik Pramanick**  
**Roll No:** 23CS60R78  
**Course:** CS69011 Computing Lab  
