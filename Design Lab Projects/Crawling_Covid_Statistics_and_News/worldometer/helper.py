from urllib.request import Request, urlopen
from collections import OrderedDict
import re, os

def page_downloader(link, path):
    if os.path.isfile(path):
        return 1
    try : 
        req = Request(link,headers ={'User-Agent':'Mozilla/5.0'})
        webpage = urlopen(req).read()
        mydata = webpage.decode("utf8")
        f=open(path,'w',encoding="utf-8")
        f.write(mydata)
        f.close
        return 1
    except : 
        return 0
    

def parsehref(s):
    final_string = "https://www.worldometers.info/coronavirus/"
    count = 0 
    for i in s:
        if i == '"':
            if count <=2: 
                count+=1
                continue
            else: 
                break
        if count == 3:
            final_string = final_string + i
    return final_string


def parseDATA(s): 
    final_string = ""
    count = 0 
    for i in s:
        if i == "'":
            if count == 0 : 
                count+=1
                continue
            else: 
                break
        if count == 1:
            final_string = final_string + i
    return final_string
