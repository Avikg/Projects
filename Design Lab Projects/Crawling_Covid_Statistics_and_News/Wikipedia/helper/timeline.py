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


global_output_file = None
###DEFINING TOKENS###
tokens = ('FIRST',
          'SECOND',
          'CONTENT',
          'SKIPTAG',
          'HEADSTART',
          'HEADEND',
          'OPENHREF',
          'CLOSEHREF',
          'OPENFIG',
          'CLOSEFIG')
t_ignore = ' \t\n'
date = ""
s = ""
x = ""
###############Tokenizer Rules################
def t_FIRST(t):
     r'<h2><span.class="mw-headline".id="Pandemic_chronology">'
     return t

def t_SECOND(t):
    r'<h2><span.class="mw-headline".id="Summary">'
    return t

def t_HEADSTART(t):
    r'<h3>'
    return t

def t_HEADEND(t):
    r'</span></h3>'
    return t

def t_OPENHREF(t):
    r'<a[^>]*>'
    return t

def t_CLOSEHREF(t):
    r'</a[^>]*>'
    return t

def t_OPENFIG(t):
    r'<figure[^>]*>'
    return t

def t_CLOSEFIG(t):
    r'</figure[^>]*>'
    return t

def t_CONTENT(t):
    r'[A-Za-z0-9, .–]+'
    return t

def t_SKIPTAG(t):
    r'<[^>]*>'
    pass

def t_error(t):
    t.lexer.skip(1)
####################################################################################################################################################################################################
											#GRAMMAR RULES
def p_start(p):
    '''start : section'''

def p_section(p):
    '''section : FIRST skipTags contentSequence SECOND '''

def p_skipTags(p):
    '''
    skipTags : CONTENT skipTags
            |  OPENHREF skipTags
            | CLOSEHREF skipTags
            |
    '''
# count = 0
def p_contentSequence(p):
    '''contentSequence : date contentElement contentSequence 
                       | contentElement
    '''
    global x 
    global s
    if len(x) > 0 and len(s)>0:
        tt = parse_dates(x, date)
        ss = chunk_string(s, 100)
        for i in tt : 
            for j in ss: 
                global_output_file.write(i + "\t" + j +"\n")
    x = ""
    s = ""

def p_date(p): 
    '''
    date : HEADSTART CONTENT OPENHREF CONTENT CLOSEHREF HEADEND
         | HEADSTART CONTENT HEADEND
    '''
    # global count
    # print(count)
    # count+=1
    global x 
    global s
    global date
    if len(x) > 0 and len(s)>0:
        tt = parse_dates(x, date)
        ss = chunk_string(s, 100)
        for i in tt : 
            for j in ss: 
                global_output_file.write(i + "\t" + j +"\n")
    x = str(p[2])
    s = ""
    



def p_contentElement(p):
    '''contentElement : OPENHREF skip CLOSEHREF contentElement
                      | OPENFIG skipcontent CLOSEFIG contentElement
                      | CONTENT contentElement
                      | empty
                      '''
    global global_output_file
    global s 
    if len(p) == 3:
        s = str(p[1]) + " " + s 

def p_skip(p):
    '''skip : CONTENT skip
            | empty'''
    global global_output_file
    global s
    if len(p) == 3:
        try:
            float(p[1])
            int(p[1])
        except ValueError:
            if p[1] != 'edit':
                s = str(p[1]) +" "+ s

def p_skipcontent(p):
    '''skipcontent : CONTENT skipcontent
                   | OPENHREF skipcontent
                   | CLOSEHREF skipcontent
                   | empty'''

def p_content(p):
    ''' content : CONTENT'''
    global s
    if len(p) == 2:
        s = str(p[1]) + " " + s

def p_error(p):
    pass

def p_empty(p):
    '''empty :'''
    pass


#########DRIVER FUNCTION#######
def process_html_page(html_page_name, output_dir, year):
    global global_output_file
    global date 
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
        global_output_file.close() 

def crawls():
    timeline_file = './Pages/timeline.txt'
    output_dir = os.path.join('Pages', 'timeline_data')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(timeline_file, 'r', encoding="utf-8") as file:
        for line in file:
            html_page_name, date  = line.strip().split() 
            process_html_page(html_page_name, output_dir, date)