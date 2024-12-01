import ply.lex as lex 
import ply.yacc as yacc
from urllib.request import Request, urlopen
import re
import os
from . import helper


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###DEFINING TOKENS###
tokens = ('STARTER','SPACE','GARBAGE', 'CONTENT', 'OPENBRACKET', 'CLOSEBRACKET', 'DATA')
t_ignore = '\t'

my_dict = {}
###############Tokenizer Rules################


def t_STARTER(t):
    r"name:.'Cases'|name:.'Daily.Deaths'|name:.'Daily.Cases'|name:.'New.Recoveries'"
    return t

def t_DATA(t): 
    r'data:.\['
    return t

def t_GARBAGE(t):
    r'<[^>]*>'

def t_SPACE(t):
    r'\s+'

def OPENBRACKET(t):
    r'\['
    return t

def CLOSEBRACKET(t):
    r'\]'
    return t

def t_CONTENT(t):
    r"[a-z0-9, ]+"
    return t

def t_error(t):
    t.lexer.skip(1)
####################################################################################################################################################################################################
											#GRAMMAR RULES

def p_start(p):
    '''
    start : STARTER skipCONTENT DATA CONTENT
    '''
    global my_dict
    s = helper.parseDATA(str(p[1])).strip().lower()
    my_dict[s] = str(p[4]).strip().split(',')



def p_skipCONTENT(p): 
    '''
    skipCONTENT : CONTENT skipCONTENT
                |  
    '''
def p_error(p):
    pass


def extract_info(file_path): 
    global lists
    global l 
    l = []
    lists = []
    file_obj= open(file_path,'r',encoding="utf-8")
    dat=file_obj.read()
    lexer = lex.lex()
    lexer.input(dat)
    parser = yacc.yacc()
    parser.parse(dat)
    for k, v in my_dict.items(): 
        for i in range(len(v)):
            if v[i] == 'null': 
                v[i] = '0' 
    return my_dict 


if __name__ == '__main__':
    extract_info("HTML\india.html")