import sys 

country = None 
date = None 
cases = 0 
Country = sys.argv[1]
Country = Country.strip().lower()
my_dict = {}


for line in sys.stdin: 
    date, country, cases = line.strip().split()
    country = country.strip().lower()
    try: 
        cases = int(cases)
    except: 
        print("Error as the space seperated strings cannot be treated as Integers")
        continue
    if country in my_dict:
        if my_dict[country] == 0 : 
            my_dict[country] = 1
        my_dict[country] = (100*(cases - my_dict[country])/my_dict[country])
    else : 
        my_dict[country] = cases


print(f"{'Decrease' if my_dict[Country]<0 else 'Increase'} in {sys.argv[2]} {sys.argv[3]} for {Country} : {abs(my_dict[Country])}%")
difference = 99999999999
similar = ""
Percentage = 0 
for k, v in my_dict.items(): 
    if k == Country : 
        continue
    if abs(v - my_dict[Country]) < difference : 
        similar = k 
        difference = abs(v-my_dict[Country])
        Percentage = v
# 
print(f"Closest Country with Similar Statistics is {similar} with {Percentage}%")
