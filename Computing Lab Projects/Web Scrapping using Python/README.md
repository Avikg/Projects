# Web Scraping System with Python (CS69011)

## Overview

This repository contains solutions for **Assignment: Web Scraping System with Python** as part of the **CS69011 Computing Lab** course. The assignment is divided into **Part A** and **Part B**, focusing on collecting, processing, and storing structured and unstructured data using Python, with optimizations through multiprocessing.

---

## Assignment Structure

```
.
├── PartA/
│   ├── 23CS60R78_Assgn_6_1.py      # Weather data scraper
│   ├── 23CS60R78_Assgn_6_2.py      # Summer Olympics data scraper
│   ├── Weather.db                  # SQLite database for weather data
│   ├── OlympicsData.db             # SQLite database for Summer Olympics data
│   ├── 23CS60R78_Assgn_6_2.txt     # Output summary for Summer Olympics scraper
├── PartB/
│   ├── 23CS60R78_Assgn_6_3.py      # Multiprocessing handler
│   ├── scraper.py                  # Worker process for web scraping
│   ├── checker.py                  # Checker script for database validation
│   ├── OlympicsData.db             # SQLite database for Summer Olympics (shared with Part A)
│   ├── 23CS60R78_Assgn_6_3.txt     # Summary of multiprocessing speedup experiment
└── README.md                       # This file
```

---

## Libraries Required

Install the necessary libraries using pip:

```bash
pip install requests urllib3 bs4 sqlite3
```

---

## Tasks

### Part A

#### Problem 1: Collecting and Storing Structured JSON Data
- **Objective**: Fetch weather data for multiple cities using the OpenWeatherMap API and store it in a SQLite database.
- **Steps**:
  1. Collect weather data (e.g., city, temperature, weather description, humidity, wind speed).
  2. Store data in the `Weather.db` database with a table named `city_weather`.

#### Problem 2: Collecting and Processing Unstructured Data
- **Objective**: Scrape the Summer Olympics Wikipedia page and store extracted data in a SQLite database.
- **Steps**:
  1. Parse the Summer Olympics Wikipedia page.
  2. Randomly select two Olympics from the past 50 years and extract relevant data (e.g., participating nations, athletes, sports, top 3 countries).
  3. Store the data in `OlympicsData.db` with a table named `SummerOlympics`.

---

### Part B

#### Problem 3: Using Multiple Processes for Speed-Up
- **Objective**: Optimize web scraping tasks using multiprocessing.
- **Steps**:
  1. Write a handler function to populate the `OlympicsData.db` with 10 Summer Olympics URLs.
  2. Spawn three processes (`scraper.py`) to scrape and populate data for each Olympic event.
  3. Use `checker.py` to validate database population and answer queries.
  4. Record speedup results and document the experiment.

---

## Database Schema

### Weather.db
- Table: `city_weather`
  - `City`: Text
  - `Temperature`: Float
  - `Description`: Text
  - `Humidity`: Integer
  - `WindSpeed`: Float

### OlympicsData.db
- Table: `SummerOlympics`
  - `Name`: Text
  - `WikipediaURL`: Text
  - `Year`: Integer
  - `HostCity`: Text
  - `ParticipatingNations`: Text
  - `Athletes`: Integer
  - `Sports`: Text
  - `Rank_1_nation`: Text
  - `Rank_2_nation`: Text
  - `Rank_3_nation`: Text
  - `DONE_OR_NOT_DONE`: Integer (1 or 0)

---

## Usage

### Part A
1. **Run the Weather Scraper**:
   ```bash
   python PartA/23CS60R78_Assgn_6_1.py
   ```
2. **Run the Summer Olympics Scraper**:
   ```bash
   python PartA/23CS60R78_Assgn_6_2.py
   ```

### Part B
1. **Run the Handler**:
   ```bash
   python PartB/23CS60R78_Assgn_6_3.py
   ```
2. **Check Database and Answer Queries**:
   ```bash
   python PartB/checker.py
   ```

---

## Experiment Report

Part B includes an experiment to measure the percentage speedup achieved using multiprocessing. Results are documented in `23CS60R78_Assgn_6_3.txt`.

---

## Author

**Avik Pramanick**  
**Roll No:** 23CS60R78  
**Course:** CS69011 Computing Lab  
