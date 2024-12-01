import ply.lex as lex
import ply.yacc as yacc
from urllib.request import Request, urlopen
import re
import os
from . import helper


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###DEFINING TOKENS###
tokens = ('STARTER','SPACE','YESTERDAY','BORN', 'HREF','GARBAGE', 'OPENTD', 'CLOSETD', 'CONTENT','TBD','CTBD','TR','CTR')
t_ignore = '\t'

cap = ""
l = []
lists = []

###############Tokenizer Rules################

def t_YESTERDAY(t):
    r'<div.class="tab-pane.".id="nav-yesterday".role="tabpanel".aria-labelledby="nav-yesterday-tab">'
    return t 

def t_STARTER(t):
    r'<td.style="display:none".data-continent="all">All</td>'
    return t

def t_HREF(t):
    r'<a[^>]*>'
    return t

def t_TBD(t):
    r'<tbody[^>]*>'
    # return t

def t_CTBD(t):
    r'</tbody[^>]*>'
    return t

def t_TR(t):
    r'<tr[^>]*>'
    return t

def t_CTR(t):
    r'</tr[^>]*>'
    return t

def t_OPENTD(t):
    r'<td[^>]*>'
    # return t

def t_CLOSETD(t):
    r'</td[^>]*>'
    # return t

def t_GARBAGE(t):
    r'<[^>]*>'

def t_SPACE(t):
    r'\s+'

def t_CONTENT(t):
    r"[A-Za-z0-9, /.;&-]+"
    # return t

def t_error(t):
    t.lexer.skip(1)
####################################################################################################################################################################################################
											#GRAMMAR RULES

def p_start(p):
    '''
    start : YESTERDAY skipGARBAGE STARTER skipTD rows
    '''
def p_skipTD(p):
    '''
    skipTD : CTR skipTD
           | 
    '''

def p_rows(p): 
    '''
    rows : GETrows rows
         | CTBD
    '''

def p_GETrows(p): 
    '''
    GETrows : TR HREF HREF CTR 
            | TR HREF CTR 
            | TR CTR
    '''
    global count 
    global lists
    if len(p) > 3 : 
        s = helper.parsehref(p[2])
        lists.append(s)
    else :
        s = " -- "
        lists.append(s)

def p_getCONTENTS(p):
    '''
    getCONTENTS : OPENTD getCONTENTS
                | OPENTD CONTENT getCONTENTS 
                |
    '''
    global l 
    if len(p) == 4 :
        l.append(str(p[2]).strip())
    if len(p) == 3 :
        l.append("0")


def p_skipGARBAGE(p):
    '''
    skipGARBAGE : TR skipGARBAGE 
            | CTR skipGARBAGE
            | 
    '''

def p_error(p):
    pass


def extract_urls(): 
    url = "https://www.worldometers.info/coronavirus/"
    page = "coronavirus"
    main_path = "./HTML/" + page + ".html"
    if not os.path.exists(main_path):
        helper.page_downloader(url, main_path)
    global lists
    global l 
    l = []
    lists = []
    file_obj= open(main_path,'r',encoding="utf-8")
    dat=file_obj.read()
    lexer = lex.lex()
    lexer.input(dat)
    parser = yacc.yacc()
    parser.parse(dat)
    return lists


# if __name__ == '__main__':
#     extract_table("result_1a.txt")