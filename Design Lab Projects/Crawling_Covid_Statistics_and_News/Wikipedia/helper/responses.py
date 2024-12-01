import os
import ply.lex as lex
import ply.yacc as yacc
from datetime import datetime, timedelta

def parse_dates(date_str, year):
    date_str = date_str.replace("–", "-")
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)
            
    try :
        if not date_str[0].isdigit(): 
            tt = date_str.strip().split(" ")
            date_str = f"{tt[1]} {tt[0]}"
        if '-' in date_str:
            day_start, day_end, month_name = date_str.replace(" ", "").partition("-")[0], date_str.split("-")[1].split(" ")[0], date_str.split(" ")[-1]
            month = datetime.strptime(month_name, "%B").month  
            start_date = datetime.strptime(f"{year}-{month}-{day_start}", "%Y-%m-%d")
            end_date = datetime.strptime(f"{year}-{month}-{day_end}", "%Y-%m-%d")
            dates = [single_date.strftime("%Y-%m-%d") for single_date in daterange(start_date, end_date)]
        else:
            day, month_name = date_str.split(" ")
            month = datetime.strptime(month_name, "%B").month 
            date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
            dates = [date.strftime("%Y-%m-%d")]
            
        return dates
    except: 
        print(date_str)
        return []
def chunk_string(s, chunk_size):
    return [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]

has_seen_reference = False
global_output_file = None
###DEFINING TOKENS###
tokens = ('FIRST',
          'SECOND',
          'CONTENT',
          'SKIPTAG',
          'OPENHREF',
          'CLOSEHREF',
          'H3')
t_ignore = ' \t\n'
edit_encounter = 0 
s = ""
x = ""
date = ""
###############Tokenizer Rules################
def t_FIRST(t):
     r'<h2>'
     return t

def t_H3(t): 
    r'<h3[^>]*>'
    return t

def t_OPENHREF(t):
    r'<a[^>]*>'
    return t

def t_CLOSEHREF(t):
    r'</a[^>]*>'
    return t

def t_CONTENT(t):
    r'[A-Za-z0-9, .\-:\'()]+'
    return t

def t_SKIPTAG(t):
    r'<[^>]*>'
    pass

def t_error(t):
    t.lexer.skip(1)
####################################################################################################################################################################################################
											#GRAMMAR RULES
def p_start(p):
    '''start : content_section'''

def p_content_section(p):
    '''content_section : FIRST content_items'''

count = 1
def p_content_items(p):
    '''content_items : content_item content_items
                     | content_item'''


def p_content_item(p):
    '''content_item : contents
                    | FIRST
                    | link
                    | heading'''

def p_heading(p):
    '''
    heading : H3 CONTENT
    '''
    global x 
    global s
    if len(x) and len(s) : 
        tt = parse_dates(x, date)
        ss = chunk_string(s, 100)
        for i in tt : 
            for j in ss: 
                global_output_file.write(i + "\t" + j +"\n")
    s = ""
    x = str(p[2])


def p_contents(p):
    '''contents : CONTENT'''
    global has_seen_reference 
    global global_output_file
    global s 
    if p[1] == "See also":
        has_seen_reference = True

    if not has_seen_reference and p[1] != '160':
        s+=(str(p[1]))

def p_link(p):
    '''link : OPENHREF CONTENT CLOSEHREF
            | OPENHREF CONTENT CONTENT CONTENT CLOSEHREF
            | OPENHREF CONTENT CONTENT CLOSEHREF
            | OPENHREF CLOSEHREF'''
    global has_seen_reference 
    global global_output_file
    global s 
    global x
    global edit_encounter
    if len(p) == 4:
        try:
            float(p[2])
            int(p[2])
        except ValueError:
            if p[2] != 'edit' and not has_seen_reference:
                s+=(str(p[2]))
    if len(p) == 5 :
        try:
            float(p[2])
            int(p[2])
            float(p[3])
            int(p[3])
        except ValueError:
            if p[2] != 'edit' and p[3] != 'edit' and not has_seen_reference:
                s+=(str(p[2]))
                s+=(str(p[3]))



def p_error(p):
    pass

#########DRIVER FUNCTION#######
def process_html_page(html_page_name, output_dir, year):
    global has_seen_reference, global_output_file
    global date
    has_seen_reference = False
    date = year
    output_file_path = os.path.join(output_dir, os.path.basename(html_page_name).replace('.html', '.txt'))
    
    global_output_file = open(output_file_path, 'w', encoding="utf-8")
    file_obj = open(html_page_name, 'r', encoding="utf-8")
    data = file_obj.read()
    file_obj.close()
    
    with open(output_file_path, 'w+', encoding="utf-8") as output_file:
        lexer = lex.lex()
        lexer.input(data)
        
        parser = yacc.yacc()
        parser.parse(data)
        global x 
        global s
        if len(x) and len(s) :
            tt = parse_dates(x, date)
            ss = chunk_string(s, 100)
            for i in tt : 
                for j in ss: 
                    global_output_file.write(i + "\t" + j +"\n")
        global_output_file.close() 

def crawls():
    responses_file = './Pages/responses.txt'
    output_dir = os.path.join('Pages', 'responses_data')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(responses_file, 'r', encoding="utf-8") as file:
        for line in file:
            html_page_name , year = line.strip().split()
            process_html_page(html_page_name, output_dir, year)