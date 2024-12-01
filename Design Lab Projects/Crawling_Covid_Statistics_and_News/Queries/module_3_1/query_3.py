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
        while True :
            choice = input("\nChoose an option(1-2)(Other Numbers/Texts to Exit):\n"
                            "1. News in given range \n"
                            "2. Responses in given range \n"
                            "Enter the corresponding number (1-2): "
                        )
            if not choice.isdigit() and not (1 <= int(choice) <= 2) :
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
                folder_path = "../../wikipedia/Pages/timeline_data"
                file_names = ["../../wikipedia/Pages/timeline_data/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

                first_command = f"python mapper.py {file_names[0]} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    command = f"sort | python combiner.py {start_date} {end_date} && python mapper.py {file_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                last_command = f"sort | python combiner.py {start_date} {end_date} | sort | python reducer.py {start_date} {end_date} {'News.txt'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()

                # Print the output if needed
                print("\n" + output.decode())
            elif choice == '2' : 
                folder_path = "../../wikipedia/Pages/responses_data"
                file_names = ["../../wikipedia/Pages/responses_data/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

                first_command = f"python mapper.py {file_names[0]} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    command = f"sort | python combiner.py {start_date} {end_date} && python mapper.py {file_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                last_command = f"sort | python combiner.py {start_date} {end_date} | sort |  python reducer.py {start_date} {end_date} {'Responses.txt'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()

                # Print the output if needed
                print("\n" + output.decode())
menu()