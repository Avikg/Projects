# import PySimpleGUI as sg
from . import worldometer_menu 
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from . import helper
from . import graph_data

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


# GUI Layout
layout = [
    [sg.Text('Enter Country Name:'), sg.InputText(key='-COUNTRY-')],
    [sg.Button('Extract Data From Table'), sg.Button('Extract Data From Graph')],
    [sg.Text(size=(60,2), key='-OUTPUT-')],
    [sg.Exit()]
]

# Create the Window
window = sg.Window('Data Extraction', layout)

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

# Event Loop
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
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
