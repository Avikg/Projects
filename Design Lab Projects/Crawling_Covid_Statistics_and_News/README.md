
# Lex Yacc and NoSQL - crawling Covid Statistics and News

## Overview

This project provides a comprehensive toolkit for accessing and analyzing data from Worldometer and Wikipedia, with a focus on tracking and visualizing various statistics related to global health and information trends. It features a command-line interface for easy navigation and data manipulation, as well as a graphical user interface (GUI) for a more interactive experience.

## Installation(Important)

To set up the project environment, ensure Python is installed on your system. This project uses `python` command for execution, rather than `python3`. Ensure all required dependencies are installed by running:

```
pip install PySimpleGUI
pip install ply
```
## Part 1: Worldometer

### Main File

- **File Name:** `worldometer_menu.py`

### Description

Running `worldometer_menu.py` grants access to the Worldometer menu, facilitating interactions with the Worldometer data through two main functions:

1. **run()**: Launches the menu in the command-line terminal, allowing users to navigate through different options.
2. **generate_files()**: Generates and stores the crawled data into five categories, organizing the information for further analysis.

### Data Storage

The data is stored in the following structure:

- **data folder**: Contains the main table country-wise in separate text files.
- **active_cases folder**: Stores the graph data of all the active cases country-wise.
- **daily_deaths folder**: Stores the graph data of all the daily deaths country-wise.
- **new_cases folder**: Stores the graph data of all the new cases country-wise.
- **new_recovered folder**: Stores the graph data of all the recovered cases country-wise.

### Usage

To use the Worldometer features, run:

```
python worldometer_menu.py
```

## Part 2: Wikipedia

### Description

Run the `main()` function to extract all the timeline and responses data from Wikipedia. The extracted data is structured as follows:

```
YYYY-MM-DD \t 100 characters
```

## Part 3: Queries

### Description

The project is designed to handle queries stored inside three separate folders, each dedicated to a specific module:

1. **module 1**: Handles the queries related to the Worldometer data.
2. **module 2**: Handles queries related to percentage change in numbers.
3. **module 3**: Handles queries related to news and responses.

## Part 4: GUI

### Description

For a graphical user interface experience, from the root folder run:

```
python -m queries_gui.py
```

This launches the GUI, allowing users to interact with the query system in a more user-friendly manner.

