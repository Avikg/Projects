from ortools.linear_solver import pywraplp

def read_input(input_file):
    N = int(input_file.readline().strip())
    M = int(input_file.readline().strip())
    profits = list(map(int, input_file.readline().strip().split()))
    resources_avail = list(map(int, input_file.readline().strip().split()))
    resource_usage = [list(map(int, input_file.readline().strip().split())) for _ in range(N)]
    return N, M, profits, resources_avail, resource_usage

def solve_production(N, M, profits, resources_avail, resource_usage):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    
    if not solver:
        return None

    production = [solver.NumVar(0, solver.infinity(), f'Product_{i}') for i in range(N)]

    objective = solver.Objective()
    for i in range(N):
        objective.SetCoefficient(production[i], profits[i])
    objective.SetMaximization()

    for j in range(M):
        constraint = solver.Constraint(0, resources_avail[j])
        for i in range(N):
            constraint.SetCoefficient(production[i], resource_usage[i][j])

    solver.Solve()

    return solver, production, objective

def write_output(output_file, solver, production, objective):
    with open(output_file, "w") as output:
        if solver.Solve() == pywraplp.Solver.OPTIMAL:
            output.write("Optimal production plan found:\n")
            for i, var in enumerate(production):
                output.write(f"Product {i}: Quantity = {var.solution_value()}\n")
            output.write(f"Maximum Profit: {objective.Value()}\n")
        else:
            output.write("The problem does not have an optimal solution.\n")

if __name__ == "__main__":
    input_file_path = 'input.txt'
    output_file_path = 'output.txt'

    with open(input_file_path, 'r') as input_file:
        N, M, profits, resources_avail, resource_usage = read_input(input_file)

    solver, production, objective = solve_production(N, M, profits, resources_avail, resource_usage)

    if solver:
        write_output(output_file_path, solver, production, objective)
