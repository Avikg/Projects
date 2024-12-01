import re
import ply.lex as lex
import ply.yacc as yacc

###DEFINING TOKENS###
tokens = ('FIRST',
          'SECOND',
          'CONTENT',
          'SKIPTAG',
          'START',
          'LAST')
t_ignore = '\t'

###############Tokenizer Rules################
def t_START(t):
    r'<span.class="mw-headline".id="Worldwide_timelines_by_month_and_year">'
    return t

def t_FIRST(t):
     r'<a[^>]*>'
     return t

def t_SECOND(t):
    r'</a[^>]*>'
    return t

def t_LAST(t):
    r'<h2>'
    return t

def t_CONTENT(t):
    r'[A-Za-z0-9, -]+'
    return t

def t_SKIPTAG(t):
    r'<[^>]*>'
    return t

def t_error(t):
    t.lexer.skip(1)
####################################################################################################################################################################################################
											#GRAMMAR RULES
parsing_scope = False
count = 0

def p_start(p):
    '''start : table'''
my_dict = {}
def p_dataCell(p):
    '''dataCell : FIRST CONTENT CONTENT SECOND
                | FIRST CONTENT CONTENT CONTENT SECOND '''
    global parsing_scope
    global count
    global my_dict
    if parsing_scope and len(p) == 5:
        if any(country in p[2] for country in ('India', 'Australia', 'Malaysia', 'England', 'Singapore')):
            if count <= 63:
                link = re.search(r'href="([^"]+)"', p[1])
                if link:
                    my_dict[link.group(1)] = 1
                count += 1
    if parsing_scope and len(p) == 6:
        if any(country in p[2] for country in ('India', 'Australia', 'Malaysia', 'England', 'Singapore')):
            if count <= 63:
                link = re.search(r'href="([^"]+)"', p[1])
                if link:
                    my_dict[link.group(1)] = 1
                count += 1

def p_table(p):
    '''table : startTag tableContent lastTag
             | tableContent'''
    pass

def p_tableContent(p):
    '''tableContent : dataCell tableContent
                    | SKIPTAG tableContent
                    | CONTENT tableContent
                    | empty'''
    pass

def p_start_tag(p):
    '''startTag : START'''
    global parsing_scope
    parsing_scope = True

def p_last_tag(p):
    '''lastTag : LAST'''
    global parsing_scope
    parsing_scope = False

def p_empty(p):
    '''empty :'''
    pass

def p_content(p):
    '''content : CONTENT
               | empty'''
    p[0] = p[1]

def p_error(p):
    pass

#########DRIVER FUNCTION#######
def crawls():
    file_obj= open('./Pages/webpage.html','r',encoding="utf-8")
    data=file_obj.read()
    lexer = lex.lex()
    lexer.input(data)
    wf = open('./Pages/tokenlist.txt','w+', encoding="utf-8")
    for tok in lexer:
        wf.write(str(tok)+'\n')
    parser = yacc.yacc()
    parser.parse(data)
    file_obj.close()
    global my_dict 
    for k, v in my_dict.items(): 
        with open('./Pages/extracted_country_link.txt', 'a+',encoding="utf-8") as link_file:
            link_file.write(k+'\n')