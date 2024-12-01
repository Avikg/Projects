from worldometer import worldometer_menu
from Wikipedia import main

worldometer_menu.generate_files('./Queries/worldometers_countrylist.txt')
main.run()

import PySimpleGUI as sg
import subprocess
import os
from datetime import datetime
from worldometer import worldometer_menu 
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from worldometer import helper
from worldometer import graph_data



# pre_processing :
# ./Queries/worldometers_countrylist.txt
folder_path = './worldometer/Results/data' 

# Get all file names in the folder
file_names = ['./worldometer/Results/data/'+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


input_mappings = [2,8,4,6,12,11,13,3,5,7]
input_choice_1 = [  "Total cases",
                    "Active cases",
                    "Total deaths",
                    "Total recovered",
                    "Total tests",
                    "Death/million",
                    "Tests/million",
                    "New case",
                    "New death",
                    "New recovered"]
country_dict = worldometer_menu.run()
country_mappings = {} 
f = open('./worldometer/Results/table_data.txt')
for line in f : 
    line = line.strip().split('\t')
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



# GUI Layout
layout = [
    [sg.Text('Enter Country Name:'), sg.InputText(key='-COUNTRY-')],
    [sg.Button('Query Worldometer'), sg.Button('Query Wikipedia'), sg.Button('Query TimeLine-Response'),sg.Button('Extract Data From Table'), sg.Button('Extract Data From Graph')],
    [sg.Multiline(size=(45, 10), key='-OUTPUT-', autoscroll=True, disabled=True)],
    [sg.Exit()]
]

def create_worldometer_window():
    layout = [
        [sg.Text('Enter Country Name:'), sg.InputText(key='-COUNTRY-')],
        [sg.Text('Choose an option (1-10):')],
        [sg.Listbox(values=['1. Total cases',
                            '2. Active cases',
                            '3. Total deaths',
                            '4. Total recovered',
                            '5. Total tests',
                            '6. Death/million',
                            '7. Tests/million',
                            '8. New case',
                            '9. New death',
                            '10. New recovered'], size=(30, 10), key='-CHOICE2-')],
        [sg.Button('Select'), sg.Button('Cancel')]
    ]
    return sg.Window('Data Type Selection', layout)

def create_wikipedia_window():
    layout = [
        [sg.Text('Enter Country Name:'), sg.InputText(key='-COUNTRY-')],
        [sg.Text('Enter Start Date [dd-mm-yyyy format]:'), sg.InputText(key='-STARTDATE-')],
        [sg.Text('Enter End Date [dd-mm-yyyy format]:'), sg.InputText(key='-ENDDATE-')],
        [sg.Text('Choose an option (1-4):')],
        [sg.Listbox(values=["1. Change in active cases in %",
                            "2. Change in daily death in %",
                            "3. Change in new recovered in %",
                            "4. Change in new cases in %"], size=(30, 10), key='-CHOICE2-')],
        [sg.Button('Select'), sg.Button('Cancel')]
    ]
    return sg.Window('Data Type Selection', layout)


def create_response_timeline_window():
    layout = [
        [sg.Text('Enter Start Date [dd-mm-yyyy format]:'), sg.InputText(key='-STARTDATE-')],
        [sg.Text('Enter End Date [dd-mm-yyyy format]:'), sg.InputText(key='-ENDDATE-')],
        [sg.Text('Choose an option (1-2):')],
        [sg.Listbox(values=["1. News in given range",
                            "2. Responses in given range",], size=(30, 10), key='-CHOICE2-')],
        [sg.Button('Select'), sg.Button('Cancel')]
    ]
    return sg.Window('Data Type Selection', layout)

def create_table_window():
    layout = [
        [sg.Text('Choose an option (1-10):')],
        [sg.Listbox(values=['1. Total cases', '2. Active cases', '3. Total deaths', '4. Total recovered',
                            '5. Total tests', '6. Death/million', '7. Tests/million',
                            '8. New case', '9. New death', '10. New recovered'], size=(30, 10), key='-CHOICE2-')],
        [sg.Button('Select'), sg.Button('Cancel')]
    ]
    return sg.Window('Data Type Selection', layout)

def create_graphical_window():
    layout = [
        [sg.Text('Choose an option for visualization:')],
        [sg.Listbox(values=['1. Active Cases', '2. Daily Death', '3. New Recovered', '4. New Cases'], size=(20, 4), key='-GRAPH_CHOICE-')],
        [sg.Button('OK'), sg.Button('Cancel')]
    ]
    return sg.Window('Select Data for Visualization', layout, finalize=True)



# Create the Window
window = sg.Window('Data Extraction', layout)

while True : 
    event, values = window.read() 
    if event in (None, 'Exit'):
        break
    elif event == 'Query Worldometer':
        window.hide()
        window2 = create_worldometer_window()
        event2, values2 = window2.read()
        if event2 == 'Select': 
            choice2 = values2['-CHOICE2-'][0][0]
            country = values2['-COUNTRY-'].strip().lower()
            if country not in country_mappings:
                sg.popup('Invalid Country')
            if choice2.isdigit() and 1 <= int(choice2) <= 10:
                # selected_data = country_dict[country][input_mappings[int(choice2)-1]]
                # window['-OUTPUT-'].update(f'{input_choice_1[int(choice2)-1]} : {selected_data}')
                index = input_mappings[int(choice2) - 1]
                # Create the first process
                first_command = f'python .\Queries\module_1\mapper.py {file_names[0]} {index} '
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    command = f'python .\Queries\module_1\combiner.py && python .\Queries\module_1\mapper.py {file_name} {index}'
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                reducer = ".\Queries\module_1\\reducer.py"
                last_command = f'python {reducer} {country_mappings[country]}'
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()
                window['-OUTPUT-'].update(f'{output.decode()}')
            else:
                window['-OUTPUT-'].update('Invalid selection or data not available.')
        window2.close()
        window.un_hide() 
    elif event ==  'Query Wikipedia':
        window.hide()
        window2 = create_wikipedia_window()
        event2, values2 = window2.read()
        if event2 == 'Select': 
            date1 = values2['-STARTDATE-'].strip()
            start_date = validate_date(date1)
            date2 = values2['-ENDDATE-'].strip()
            end_date = validate_date(date2)
            start_date = format_date(start_date)
            end_date = format_date(end_date) 

            choice2 = values2['-CHOICE2-'][0][0]
            country = values2['-COUNTRY-'].strip().lower()
            if country not in country_mappings or start_date > end_date or not (choice2.isdigit() and 1 <= int(choice2) <= 4) :
                sg.popup('Invalid Inputs')
                break
            if int(choice2) == 1:
                folder_path = "./worldometer/Results/active_cases"
                file_names = ["./worldometer/Results/active_cases/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                country_name = os.path.splitext(os.path.basename(file_names[0]))[0]
                
                first_command = f"python .\Queries\module_2\mapper.py {file_names[0]} {country_name} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    country_name = os.path.splitext(os.path.basename(file_name))[0]
                    command = f"sort | python .\Queries\module_2\combiner.py {start_date} {end_date} && python .\Queries\module_2\mapper.py {file_name} {country_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                reducer = ".\Queries\module_2\\reducer.py"
                last_command = f"sort | python .\Queries\module_2\combiner.py {start_date} {end_date} | python {reducer} {country} {'Active Cases'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                
                output, _ = last_process.communicate()
                window['-OUTPUT-'].update(f'{output.decode()}')
            elif int(choice2) == 2:
                folder_path = "./worldometer/Results/daily_deaths"
                file_names = ["./worldometer/Results/daily_deaths/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                country_name = os.path.splitext(os.path.basename(file_names[0]))[0]
                
                first_command = f"python .\Queries\module_2\mapper.py {file_names[0]} {country_name} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    country_name = os.path.splitext(os.path.basename(file_name))[0]
                    command = f"sort | python .\Queries\module_2\combiner.py {start_date} {end_date} && python .\Queries\module_2\mapper.py {file_name} {country_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                reducer = ".\Queries\module_2\\reducer.py"
                last_command = f"sort | python .\Queries\module_2\combiner.py {start_date} {end_date} | python {reducer} {country} {'Daily Deaths'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                
                output, _ = last_process.communicate()
                window['-OUTPUT-'].update(f'{output.decode()}')
            elif int(choice2) == 3: 
                folder_path = "./worldometer/Results/new_recovered"
                file_names = ["./worldometer/Results/new_recovered/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                country_name = os.path.splitext(os.path.basename(file_names[0]))[0]
                
                first_command = f"python .\Queries\module_2\mapper.py {file_names[0]} {country_name} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    country_name = os.path.splitext(os.path.basename(file_name))[0]
                    command = f"sort | python .\Queries\module_2\combiner.py {start_date} {end_date} && python .\Queries\module_2\mapper.py {file_name} {country_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                reducer = ".\Queries\module_2\\reducer.py"
                last_command = f"sort | python .\Queries\module_2\combiner.py {start_date} {end_date} | python {reducer} {country} {'New Recovered'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                
                output, _ = last_process.communicate()
                window['-OUTPUT-'].update(f'{output.decode()}')
            elif int(choice2) == 4: 
                folder_path = "./worldometer/Results/new_cases"
                file_names = ["./worldometer/Results/new_cases/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                country_name = os.path.splitext(os.path.basename(file_names[0]))[0]
                
                first_command = f"python .\Queries\module_2\mapper.py {file_names[0]} {country_name} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    country_name = os.path.splitext(os.path.basename(file_name))[0]
                    command = f"sort | python .\Queries\module_2\combiner.py {start_date} {end_date} && python .\Queries\module_2\mapper.py {file_name} {country_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                reducer = ".\Queries\module_2\\reducer.py"
                last_command = f"sort | python .\Queries\module_2\combiner.py {start_date} {end_date} | python {reducer} {country} {'New Cases'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                
                output, _ = last_process.communicate()
                window['-OUTPUT-'].update(f'{output.decode()}')
            else:
                window['-OUTPUT-'].update('Invalid selection or data not available.')
        window2.close()
        window.un_hide() 
    elif event == 'Query TimeLine-Response':
        window.hide()
        window2 = create_response_timeline_window()
        event2, values2 = window2.read()
        if event2 == 'Select': 
            date1 = values2['-STARTDATE-'].strip()
            start_date = validate_date(date1)
            date2 = values2['-ENDDATE-'].strip()
            end_date = validate_date(date2)
            start_date = format_date(start_date)
            end_date = format_date(end_date) 

            choice2 = values2['-CHOICE2-'][0][0]
            if start_date > end_date or not (choice2.isdigit() and 1 <= int(choice2) <= 2) :
                sg.popup('Invalid Inputs')
                break
            print(choice2)
            if int(choice2) == 1:
                folder_path = "./wikipedia/Pages/timeline_data"
                file_names = ["./wikipedia/Pages/timeline_data/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

                first_command = f"python .\Queries\module_3_1\mapper.py {file_names[0]} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    command = f"sort | python .\Queries\module_3_1\combiner.py {start_date} {end_date} && python .\Queries\module_3_1\mapper.py {file_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                reducer = ".\Queries\module_3_1\\reducer.py"
                last_command = f"sort | python .\Queries\module_3_1\combiner.py {start_date} {end_date} | sort | python {reducer} {start_date} {end_date} {'News.txt'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()

                window['-OUTPUT-'].update(f'Output Available at News.txt')
            if int(choice2) == 2:
                folder_path = "./wikipedia/Pages/responses_data"
                file_names = ["./wikipedia/Pages/responses_data/"+f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

                first_command = f"python .\Queries\module_3_1\mapper.py {file_names[0]} "
                first_process = subprocess.Popen(first_command, shell=True, stdout=subprocess.PIPE)

                # Create subsequent processes in a loop, connecting input and output
                for file_name in file_names[1:]:
                    command = f"sort | python .\Queries\module_3_1\combiner.py {start_date} {end_date} && python .\Queries\module_3_1\mapper.py {file_name}"
                    process = subprocess.Popen(command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)
                    first_process = process  # Update first_process to connect subsequent processes

                # Create the last process with the reducer command
                reducer = ".\Queries\module_3_1\\reducer.py"
                last_command = f"sort | python .\Queries\module_3_1\combiner.py {start_date} {end_date} | sort | python {reducer} {start_date} {end_date} {'Responses.txt'}"
                last_process = subprocess.Popen(last_command, shell=True, stdin=first_process.stdout, stdout=subprocess.PIPE)

                # Wait for the last process to finish and get the output
                output, _ = last_process.communicate()
                window['-OUTPUT-'].update(f'Output Available at Responses.txt')
            else:
                window['-OUTPUT-'].update('Data not available.')
        window2.close()
        window.un_hide() 
    elif event == 'Extract Data From Table':
        country = values['-COUNTRY-'].strip().lower()
        if country not in country_dict:
            sg.popup('Invalid Country')
        else:
            window.hide()  # Hide the main window
            window2 = create_table_window()
            event2, values2 = window2.read()
            if event2 == 'Select':
                choice2 = values2['-CHOICE2-'][0][0]  # Get the first character of the selection, which is the number
                if choice2.isdigit() and 1 <= int(choice2) <= 10:
                    selected_data = country_dict[country][input_mappings[int(choice2)-1]]
                    window['-OUTPUT-'].update(f"{input_choice_1[int(choice2)-1]} : {selected_data}")
                else:
                    window['-OUTPUT-'].update("Invalid selection or data not available.")
            window2.close()
            window.un_hide() 
    elif event == 'Extract Data From Graph':
        country = values['-COUNTRY-'].strip().lower()
        if country not in country_dict:
            sg.popup('Invalid Country')
        else:
            page = country.replace(" ", "-")
            main_path = f"./HTML/{page}.html"
            if not os.path.exists(main_path): 
                helper.page_downloader(country_dict[country][-1], main_path)
            data = graph_data.extract_info(main_path)
            window2 = create_graphical_window()
            event2, values2 = window2.read()
            window2.close()

            if event2 == 'OK':
                choice2 = values2['-GRAPH_CHOICE-'][0][0]  
                
                if choice2.isdigit() and 1 <= int(choice2) <= 4:
                    int_choice2 = int(choice2)
                    if int_choice2 == 1 and "cases" in data:
                        active_cases = list(map(int, data['cases']))
                        start_date = datetime(2020, 2, 15)
                        num_days = len(active_cases)
                        dates = [start_date + timedelta(days=i) for i in range(num_days)]
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(dates, active_cases, marker='o', color='#FF9900', linewidth=2, markersize=8)
                        plt.xlabel('Date')
                        plt.ylabel('Total Coronavirus Active Cases')
                        plt.title('Total Active Cases')
                        plt.xticks(rotation=45)
                        plt.yscale('log')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()
                    elif int_choice2 == 2 and "daily deaths" in data:
                        active_cases = list(map(int, data["daily deaths"]))
                        start_date = datetime(2020, 2, 15)
                        num_days = len(active_cases)
                        dates = [start_date + timedelta(days=i) for i in range(num_days)]
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(dates, active_cases, marker='o', color='#FF9900', linewidth=2, markersize=8)
                        plt.xlabel('Date')
                        plt.ylabel('Total Coronavirus daily deaths Cases')
                        plt.title('Total daily deaths Cases')
                        plt.xticks(rotation=45)
                        plt.yscale('log')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()
                    elif int_choice2 == 3 and "new recoveries" in data:
                        active_cases = list(map(int, data["new recoveries"]))
                        start_date = datetime(2020, 2, 15)
                        num_days = len(active_cases)
                        dates = [start_date + timedelta(days=i) for i in range(num_days)]
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(dates, active_cases, marker='o', color='#FF9900', linewidth=2, markersize=8)
                        plt.xlabel('Date')
                        plt.ylabel('Total Coronavirus new recoveries')
                        plt.title('Total new recoveries')
                        plt.xticks(rotation=45)
                        plt.yscale('log')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()
                    elif int_choice2 == 4 and "daily cases" in data:
                        active_cases = list(map(int, data["daily cases"]))
                        start_date = datetime(2020, 2, 15)
                        num_days = len(active_cases)
                        dates = [start_date + timedelta(days=i) for i in range(num_days)]
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(dates, active_cases, marker='o', color='#FF9900', linewidth=2, markersize=8)
                        plt.xlabel('Date')
                        plt.ylabel('Total Coronavirus Daily.Cases')
                        plt.title('Total Daily Cases')
                        plt.xticks(rotation=45)
                        plt.yscale('log')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()
window.close()
