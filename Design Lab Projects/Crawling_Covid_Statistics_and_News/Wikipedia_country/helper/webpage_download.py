import os
import ply.lex as lex
import ply.yacc as yacc
from urllib.request import Request, urlopen
def web_dwnload():
    req = Request('https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic',headers ={'User-Agent':'Mozilla/5.0'})
    webpage = urlopen(req).read()
    mydata = webpage.decode("utf8")
    f=open('./Pages/webpage.html','w',encoding="utf-8")
    f.write(mydata)
    f.close()
