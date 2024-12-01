## Author-Avik Pramanick, 
## Roll-23CS60R78.

# Web Crawling and Information Extraction: COVID-19 Data

This project is designed to extract and visualize COVID-19 data from various countries and continents. It is developed in Python and tested on Linux, ensuring compatibility with Unix-like operating systems.

## Requirements

- **Operating System**: Linux
- **Programming Language**: Python 3
- **External Libraries**: matplotlib, ply

Ensure Python 3 and matplotlib are installed on your system to run this project successfully.

## Installation

Clone this repository or download the source code to your local machine. Ensure `task1.py` and `task2.py` are located in the same directory since `task2.py` imports `task1.py`.

## Running the Project

To start the project, navigate to the directory containing the scripts and run the following commands in your terminal:

1. For web crawling and data extraction:
   ```
   python3 task1.py
   ```
   Ensure you have an active internet connection for `task1.py` to function correctly.

2. For processing and visualization:
   ```
   python3 task2.py
   ```

## User Guide

### Basic Commands

- To exit the program, type `exit`.
- To return to the previous menu, type `back`.
- For initial menu selection, you can use the menu name, its lowercase equivalent, or the menu number.

### Selection Menus

- **Country/Continent Selection**: Input can be the name (case-insensitive) or the number corresponding to the country or continent.
- **Information Type Selection**: Specify the exact name or number of the information type you're interested in (e.g., Total cases, Active cases).
- **Query Type Selection**: Input should be the number corresponding to the query type.
- **Query Date Range**: Input dates in the format `dd-mm-yyyy` or `d-m-yyyy`.

## Generating Data Files

To generate data files for visualization, run `main.py` and use the following command within the Python environment:

```python
import worldometer_menu
worldometer_menu.generate_files('country_list.txt')
```

This will generate data files in the respective folders for active cases, daily deaths, new cases, and new recovered cases.

## Additional Tools

A `test.py` script is included to demonstrate the functionality of the data file generation process.
