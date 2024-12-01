def parse_update_log(filename):
    updates = {}
    with open(filename, 'r') as f:
        for line in f:
            try:
                parts = line.split()
                action = parts[0]
                nodes = tuple(map(int, parts[1][1:-1].split(',')))
                updates[nodes] = action
            except Exception as e:
                print(f"Error parsing line: {line.strip()}")
    return updates

def parse_path_found_log(filename):
    paths = {'PATH_FOUND': [], 'PATH_REMOVED': []}
    with open(filename, 'r') as f:
        for line in f:
            try:
                parts = line.split()
                action = parts[0]
                nodes = tuple(map(int, parts[1][1:-1].split(',')))
                paths[action].append(nodes)
            except Exception as e:
                print(f"Error parsing line: {line.strip()}")
    return paths

def test_correctness():
    updates = parse_update_log('update.log')
    paths = parse_path_found_log('path_found.log')

    for action, node_list in paths.items():
        for nodes in node_list:
            if nodes not in updates or updates[nodes] != action:
                print(f"Error: {action} for {nodes} but no corresponding edge update found.")
    
    # Add any other checks or logic as needed

test_correctness()