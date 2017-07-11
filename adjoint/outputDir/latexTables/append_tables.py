import sys

def append_tables(append_file, table_file):

    a_file = open(append_file,'a')
    t_file = open(table_file,'r')

    a_file.write('\\\\ \n')
    a_file.write('''\\\\ \n''')
    name_split = table_file[:-4].split('_')
    name_string = ''
    for s in name_split:
        name_string += ' '+s
    name_string +=':'+'\n'
    
    a_file.write(name_string)
    a_file.write('''\\\\ \n''')

    for line in t_file.readlines():
        a_file.write(line)
    

    t_file.close()
    a_file.close()

def main():
    try:
       a_file = sys.argv[1]
       files = sys.argv[2:]
    except:
        print 'input plz'
        return

    for f in files:
        append_tables(a_file,f)


        
if __name__ == '__main__':
    main()
