import os
from urllib.request import Request, urlopen
def extract_date(s):
    last_part = s[-6:]
    year_str = last_part.replace(")", "")
    year = int(year_str)
    return year

def download_webpage(url, year):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        webpage = urlopen(req).read()
        mydata = webpage.decode("utf8")
        page_name = url.split('/')[-1]
        filename = f"{page_name}.html".replace("%", "_").replace("/", "_").replace(":", "_")
        filename = f"./Pages/"  + filename
        with open(filename, 'w+', encoding="utf-8") as f:
            f.write(mydata)

        with open('./Pages/extracted_country_data.txt', 'a+', encoding="utf-8") as f:
            f.write(f"{filename} {year}\n")
        
    except Exception as e:
        print(f"Error downloading {page_name}: {e}")

def download_from_extracted_links():
    base_url = 'https://en.wikipedia.org'
    if os.path.exists('./Pages/extracted_country_link.txt'):
        with open('./Pages/extracted_country_link.txt', 'r', encoding="utf-8") as file:
            for url_suffix in file.readlines():
                full_url = base_url + url_suffix.strip()
                year = extract_date(url_suffix)
                download_webpage(full_url, year)
    else:
        print("extracted_country_link.txt file not found.")
