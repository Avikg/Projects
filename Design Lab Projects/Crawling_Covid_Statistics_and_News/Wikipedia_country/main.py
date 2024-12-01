from helper import webpage_download
from helper import country
from helper import country_webpages
from helper import countrydata
import os

if not os.path.exists("./Pages"):
    os.mkdir("./Pages")
webpage_download.web_dwnload()
country.crawls()
country_webpages.download_from_extracted_links()
countrydata.crawl_country()