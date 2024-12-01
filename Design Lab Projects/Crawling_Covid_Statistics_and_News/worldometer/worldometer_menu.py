from . import graph_data
from . import url_fetcher
from . import table_data
import os 
from . import helper
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def menu(my_dict): 
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
    print("------------------------------------------")
    while True: 
        choice = input("Enter Choice(Corresponding Numbers 1-2):\n1.Extract Data From table\n2.Extract Data From Graph\n3.Enter 0 to exit").strip()
        while choice == '1' : 
            country = input("Enter Country Name(-1 to exit) : ").strip().lower()
            if country == '-1' :
                break
            if country not in my_dict :
                print("Invalid Country")
                continue
            else :
                while choice == '1': 
                    choice2 = input(
                                        "Choose an option(1-10)(Other Numbers/Texts to Exit):\n"
                                        "1. Total cases\n"
                                        "2. Active cases\n"
                                        "3. Total deaths\n"
                                        "4. Total recovered\n"
                                        "5. Total tests\n"
                                        "6. Death/million\n"
                                        "7. Tests/million\n"
                                        "8. New case\n"
                                        "9. New death\n"
                                        "10. New recovered\n"
                                        "Enter the corresponding number (1-10): "
                                    )
                    if choice2.isdigit() and 1 <= int(choice2) <= 10:
                        print(f"{input_choice_1[int(choice)-1]} : {my_dict[country][input_mappings[int(choice2)-1]]}")
                    else : 
                        choice = -1
        while choice == '2': 
            country = input("Enter Country Name(-1 to exit) : ").strip().lower()
            if country == '-1' :
                break
            if country not in my_dict :
                print("Invalid Country")
                continue
            else :
                page = country.replace(" ", "-")
                main_path = "./HTML/" + page + ".html"
                if not os.path.exists(main_path): 
                    helper.page_downloader(my_dict[country][-1], main_path)
                data = graph_data.extract_info(main_path)
                while choice == '2':
                    choice2 = input(
                                        "Choose an option(1-4)(Other Numbers/Texts to Exit):\n"
                                        "1. Active Cases\n"
                                        "2. Daily Death\n"
                                        "3. New Recovered\n"
                                        "4. New cases\n"
                                        "Enter the corresponding number (1-4): "
                                    )
                    if choice2.isdigit() and 1 <= int(choice2) <= 4:
                        if int(choice2) == 1: 
                            if "cases" in data : 
                                active_cases = list(map(int, data['cases']))

                                # Start date
                                start_date = datetime(2020, 2, 15)

                                # Number of days
                                num_days = len(active_cases)  # Update this based on your data

                                # Generate dates
                                dates = [start_date + timedelta(days=i) for i in range(num_days)]

                                # Create a line chart
                                plt.plot(dates, active_cases, marker='o', color='#FF9900', linewidth=2, markersize=8)

                                # Adding labels and title
                                plt.xlabel('Date')
                                plt.ylabel('Total Coronavirus Active Cases')
                                plt.title('Total Active Cases')

                                # Rotate x-axis labels for better readability
                                plt.xticks(rotation=45)
                                plt.yscale('log')
                                # Show plot
                                plt.grid(True)
                                plt.tight_layout()
                                plt.show()

                        elif int(choice2) == 2: 
                            if "daily deaths" in data : 
                                active_cases = list(map(int, data["daily deaths"]))

                                # Start date
                                start_date = datetime(2020, 2, 15)

                                # Number of days
                                num_days = len(active_cases)  # Update this based on your data

                                # Generate dates
                                dates = [start_date + timedelta(days=i) for i in range(num_days)]

                                # Create a line chart
                                plt.plot(dates, active_cases, marker='o', color='#FF9900', linewidth=2, markersize=8)

                                # Adding labels and title
                                plt.xlabel('Date')
                                plt.ylabel('Total Coronavirus daily deaths Cases')
                                plt.title('Total daily deaths Cases')

                                # Rotate x-axis labels for better readability
                                plt.xticks(rotation=45)
                                plt.yscale('log')
                                # Show plot
                                plt.grid(True)
                                plt.tight_layout()
                                plt.show()
                        elif int(choice2) == 3: 
                            if "new recoveries" in data : 
                                active_cases = list(map(int, data["new recoveries"]))

                                # Start date
                                start_date = datetime(2020, 2, 15)

                                # Number of days
                                num_days = len(active_cases)  # Update this based on your data

                                # Generate dates
                                dates = [start_date + timedelta(days=i) for i in range(num_days)]

                                # Create a line chart
                                plt.plot(dates, active_cases, marker='o', color='#FF9900', linewidth=2, markersize=8)

                                # Adding labels and title
                                plt.xlabel('Date')
                                plt.ylabel('Total Coronavirus new recoveries')
                                plt.title('Total new recoveries')

                                # Rotate x-axis labels for better readability
                                plt.xticks(rotation=45)
                                plt.yscale('log')
                                # Show plot
                                plt.grid(True)
                                plt.tight_layout()
                                plt.show()
                        elif int(choice2) == 4: 
                            if "daily cases" in data : 
                                active_cases = list(map(int, data["daily cases"]))

                                # Start date
                                start_date = datetime(2020, 2, 15)

                                # Number of days
                                num_days = len(active_cases)  # Update this based on your data

                                # Generate dates
                                dates = [start_date + timedelta(days=i) for i in range(num_days)]

                                # Create a line chart
                                plt.plot(dates, active_cases, marker='o', color='#FF9900', linewidth=2, markersize=8)

                                # Adding labels and title
                                plt.xlabel('Date')
                                plt.ylabel('Total Coronavirus Daily.Cases')
                                plt.title('Total Daily.Cases')

                                # Rotate x-axis labels for better readability
                                plt.xticks(rotation=45)
                                plt.yscale('log')
                                # Show plot
                                plt.grid(True)
                                plt.tight_layout()
                                plt.show()
                        else : 
                            print("Invalid Chice")
                            break
                    else : 
                        choice = -1
        if choice != '1' or choice != '2' : 
            break

def generate_files(file):
    if not os.path.exists("./HTML"):
        os.mkdir("./HTML")
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    if not os.path.exists("./Results/data"):
        os.mkdir("./Results/data")
    t_data = table_data.extract_table()
    url_data =url_fetcher.extract_urls()
    for l in t_data : 
        page = l[1].replace(" ", "-")
        f=open("./Results/data/"+page+".txt",'w',encoding="utf-8")
        for i in l : 
            f.write(str(i) + "\t")
        f.close()
    country_data = {}
    for i in range(len(t_data)): 
        country_data[str(t_data[i][1]).strip().lower()] = t_data[i]
        country_data[str(t_data[i][1]).strip().lower()].append(url_data[i])
    is_country = False
    country_list = []
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            if '-' in line : 
                is_country = True
                continue
            if len(line) == 0 : 
                is_country = False
            if is_country : 
                country_list.append(line.lower())
    if not os.path.exists("./Results/active_cases"):
        os.mkdir("./Results/active_cases")
    if not os.path.exists("./Results/daily_deaths"):
        os.mkdir("./Results/daily_deaths")
    if not os.path.exists("./Results/new_recovered"):
        os.mkdir("./Results/new_recovered")
    if not os.path.exists("./Results/new_cases"):
        os.mkdir("./Results/new_cases")

    for name in country_list : 
        page = name.replace(" ", "-")
        main_path = "./HTML/" + page + ".html"
        if not os.path.exists(main_path):
            helper.page_downloader(country_data[name][-1], main_path)
        data = graph_data.extract_info(main_path)
        if "cases" in data :
            start_date = datetime(2020, 2, 15)
            path_ = "./Results/active_cases/" + page + ".txt"
            f=open(path_,'w',encoding="utf-8")
            for i, d in enumerate(data["cases"]):
                current_date = start_date + timedelta(days=i)
                f.write(str(current_date.strftime('%Y-%m-%d')) + "\t" + d)
                f.write("\n")
            f.close
        if "daily deaths" in data : 
            start_date = datetime(2020, 2, 15)
            path_ = "./Results/daily_deaths/" + page + ".txt"
            f=open(path_,'w',encoding="utf-8")
            for i, d in enumerate(data["daily deaths"]):
                current_date = start_date + timedelta(days=i)
                f.write(str(current_date.strftime('%Y-%m-%d')) + "\t" + d)
                f.write("\n")
            f.close
        if "new recoveries" in data : 
            start_date = datetime(2020, 2, 15)
            path_ = "./Results/new_recovered/" + page + ".txt"
            f=open(path_,'w',encoding="utf-8")
            for i, d in enumerate(data["new recoveries"]):
                current_date = start_date + timedelta(days=i)
                f.write(str(current_date.strftime('%Y-%m-%d')) + "\t" + d)
                f.write("\n")
            f.close
        if "daily cases" in data : 
            start_date = datetime(2020, 2, 15)
            path_ = "./Results/new_cases/" + page + ".txt"
            f=open(path_,'w',encoding="utf-8")
            for i, d in enumerate(data["daily cases"]):
                current_date = start_date + timedelta(days=i)
                f.write(str(current_date.strftime('%Y-%m-%d')) + "\t" + d)
                f.write("\n")
            f.close

def run():
    if not os.path.exists("./HTML"):
        os.mkdir("./HTML")
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    t_data = table_data.extract_table()
    url_data =url_fetcher.extract_urls()
    country_data = {}
    for i in range(len(t_data)): 
        country_data[str(t_data[i][1]).strip().lower()] = t_data[i]
        country_data[str(t_data[i][1]).strip().lower()].append(url_data[i])
    return country_data 

if __name__ == '__main__':
    if not os.path.exists("./HTML"):
        os.mkdir("./HTML")
    if not os.path.exists("./Results"):
        os.mkdir("./Results")
    t_data = table_data.extract_table()
    url_data =url_fetcher.extract_urls()
    country_data = {}
    for i in range(len(t_data)): 
        country_data[str(t_data[i][1]).strip().lower()] = t_data[i]
        country_data[str(t_data[i][1]).strip().lower()].append(url_data[i])
    menu(country_data) 