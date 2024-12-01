from ortools.linear_solver import pywraplp

def maximize_sweet_box(k, m, n, x, y, cost):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return "SCIP is not available."
    sweets = []
    for i in range(k):
        sweets.append(solver.IntVar(0, 1, f'Sweet_{i}'))

    print(sweets)

   
    objective = solver.Objective()
    for i in range(k):
        objective.SetCoefficient(sweets[i], cost[i])
    objective.SetMaximization()

    
    total_area = solver.Sum(sweets[i] * x[i] * y[i] for i in range(k))
    solver.Add(total_area <= m*n) 
    solver.Solve()

    max_profit = objective.Value()
    sizes_used = [sweets[i].solution_value() for i in range(k)]

    with open("output.txt", 'w') as out_f:
        out_f.write("Size of sweets used:\n")
        for i in range(k):
            out_f.write(f"Size of sweet {i + 1}: {int(sizes_used[i] * x[i] * y[i])}\n")
        out_f.write(f"Maximum Profit: {int(max_profit)}\n")

if __name__ == "__main__":
    with open("input.txt", "r") as file:
        k = int(file.readline())
        m, n = map(int, file.readline().split())
        x = list(map(int, file.readline().split()))
        y = list(map(int, file.readline().split()))
        cost = list(map(int, file.readline().split()))
    maximize_sweet_box(k, m, n, x, y, cost)