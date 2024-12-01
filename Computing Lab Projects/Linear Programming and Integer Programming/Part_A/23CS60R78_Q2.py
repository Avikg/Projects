from ortools.linear_solver import pywraplp

def optimize_production(N, M, profit_per_unit, resource_availability, max_capacity, resource_usage, output_file):
    solver = pywraplp.Solver.CreateSolver('GLOP')

    if not solver:
        return

    infinity = solver.infinity()
    variables = []

    
    for i in range(N):
        variables.append(solver.NumVar(0.0, max_capacity[i], f'x_{i}'))

    
    objective = solver.Objective()
    for i in range(N):
        objective.SetCoefficient(variables[i], profit_per_unit[i])
    objective.SetMaximization()

    
    for j in range(M):
        constraint = solver.Constraint(0, resource_availability[j])
        for i in range(N):
            constraint.SetCoefficient(variables[i], resource_usage[i][j])

    
    if solver.Solve() == pywraplp.Solver.OPTIMAL:
        
        with open(output_file, 'w') as file:
            file.write("Optimal production plan found:\n")
            for i in range(N):
                file.write(f"Product {i}: Quantity = {variables[i].solution_value():.1f}\n")
            file.write(f"Maximum Profit: {solver.Objective().Value():.1f}\n")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    input_file_path = 'input.txt'
    output_file_path = 'output.txt'

    with open(input_file_path, 'r') as input_file:
        N = int(input_file.readline())
        M = int(input_file.readline())
        profit_per_unit = list(map(float, input_file.readline().split()))
        resource_availability = list(map(float, input_file.readline().split()))
        max_capacity = list(map(float, input_file.readline().split()))

        resource_usage = []
        for _ in range(N):
            resource_usage.append(list(map(float, input_file.readline().split())))

    optimize_production(N, M, profit_per_unit, resource_availability, max_capacity, resource_usage, output_file_path)
