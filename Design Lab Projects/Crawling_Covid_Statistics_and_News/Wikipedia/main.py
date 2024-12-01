from .helper import webpage_download
from .helper import webpages
from .helper import timeline
from .helper import project
from .helper import responses
import os

def run():
    if not os.path.exists("./Pages"):
        os.mkdir("./Pages")
    webpage_download.web_dwnload()
    project.wiki_crawl()
    webpages.categorize_and_download()
    timeline.crawls()
    responses.crawls()
