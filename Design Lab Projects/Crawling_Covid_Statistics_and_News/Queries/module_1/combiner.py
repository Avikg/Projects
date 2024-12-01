import sys 

node_a = None 
current_count = 0

for line in sys.stdin: 
    key, val = line.strip().split()
    try: 
        key = int(key)
        val = int(val)
    except: 
        print("Error as the space seperated strings cannot be treated as Integers")
        continue
    if key == node_a:
        current_count+=val
    else :
        if node_a != None:
            print(node_a , current_count)
        current_count = val
        node_a = key

print(node_a, current_count)
