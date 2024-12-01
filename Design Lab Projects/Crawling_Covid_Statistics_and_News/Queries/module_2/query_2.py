import subprocess
import os
from datetime import datetime


country_mappings = {} 
f = open("../../worldometer/Results/table_data.txt")
for line in f : 
    line = line.strip().split("\t")
    country_mappings[str(line[1]).strip().lower()] = int(line[0])
f.close

def validate_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%d-%m-%Y')
        return date_obj
    except ValueError:
        print("Incorrect date format, please enter date in dd-mm-yyyy format.")
        return None
def format_date(date_obj):
    return date_obj.strftime('%Y-%m-%d')

def menu():
    print("************************************")
    while True :
        country = input("\nEnter Name of the Country : ")
        country = country.strip()
        if country not in country_mappings : 
            print("Invalid Country")
            break 
        while True :
            choice = input("\nChoose an option(1-10)(Other Numbers/Texts to Exit):\n"
                            "1. Change in active cases in %\n"
                            "2. Change in daily death in %\n"
                            "3. Change in new recovered in %\n"
                            "4. Change in new cases in %\n"
                            "Enter the corresponding number (1-4): "
                        )
            if not choice.isdigit() and not (1 <= int(choice) <= 10) :
                print("Invalid Choice\n")
                break
            choice == int(choice)

            date1 = input("Input Starting Date [dd-mm-yyyy format] : ")
            start_date = validate_date(date1)
            date2 = input("Input Ending   Date [dd-mm-yyyy format] : ")
            end_date = validate_date(date2)

            if start_date is None or end_date is None : 
                print("Invalid Date Format ")
                continue
            start_date = format_date(start_date)
            end_date = format_date(end_date) 
            if choice == '1' : 
                folder_path = "../../worldometer/Results/active_cases"
                file_names = ["../../worldometer/Results/active_cases/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                country_name = os.path.splitext(os.path.basename(file_names[0]))[0]
                
                first_command = f"python mapper.py {file_names[0]} {country_name} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    country_name = os.path.splitext(os.path.basename(file_name))[0]
                    command = f"sort | python combiner.py {start_date} {end_date} && python mapper.py {file_name} {country_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                last_command = f"sort | python combiner.py {start_date} {end_date} | python reducer.py {country} {'Active Cases'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()

                # Print the output if needed
                print("\n" + output.decode())

            elif choice == '2' : 
                folder_path = "../../worldometer/Results/daily_deaths"
                file_names = ["../../worldometer/Results/daily_deaths/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                country_name = os.path.splitext(os.path.basename(file_names[0]))[0]
                
                
                first_command = f"python mapper.py {file_names[0]} {country_name} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    country_name = os.path.splitext(os.path.basename(file_name))[0]
                    command = f"sort -n | python combiner.py {start_date} {end_date} && python mapper.py {file_name} {country_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                last_command = f"sort-n |  python reducer.py {country} {'Daily Deaths'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()

                # Print the output if needed
                print("\n" + output.decode())
            elif choice == '3' : 
                folder_path = "../../worldometer/Results/new_recovered"
                file_names = ["../../worldometer/Results/new_recovered/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                country_name = os.path.splitext(os.path.basename(file_names[0]))[0]
                
                
                first_command = f"python mapper.py {file_names[0]} {country_name} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    country_name = os.path.splitext(os.path.basename(file_name))[0]
                    command = f"python combiner.py {start_date} {end_date} && python mapper.py {file_name} {country_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                last_command = f"sort -n | python reducer.py {country} {'New Recovered'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()

                # Print the output if needed
                print("\n" + output.decode())
            elif choice == '4': 
                folder_path = "../../worldometer/Results/new_cases"
                file_names = ["../../worldometer/Results/new_cases/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                country_name = os.path.splitext(os.path.basename(file_names[0]))[0]
                
                
                first_command = f"python mapper.py {file_names[0]} {country_name} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    country_name = os.path.splitext(os.path.basename(file_name))[0]
                    command = f"python combiner.py {start_date} {end_date} && python mapper.py {file_name} {country_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                last_command = f"sort -n | python reducer.py {country} {'New Cases'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()

                # Print the output if needed
                print("\n" + output.decode())
            else : 
                print("Invalid Input ")
                break         
menu()
               