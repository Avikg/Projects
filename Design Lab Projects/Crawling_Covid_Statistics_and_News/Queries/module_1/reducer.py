import sys 

node_1 = None 
node_a = None
current_count = 0
total_count = 0
cases = 0 
index = sys.argv[1]

for line in sys.stdin: 
    node_a, count = line.strip().split()
    try: 
        node_a = int(node_a)
        count = int(count)
        total_count+=count
    except: 
        print("Error as the space seperated strings cannot be treated as Integers")
        continue
    if node_1 == node_a:
        current_count+=count
    else :
        if node_1 and node_1 == int(index): 
            cases = current_count
        node_1 = node_a 
        current_count = count

if node_1 == index : 
    cases = current_count 

print(f"Total Count: {cases}, Percentage : {((100*cases)/total_count)}")