import pandas as pd


def NequalNum_sort(l):
    
    sort = []
    numbers = []
    for i in range(len(l)):
        
        s = l[i].split()
        I = int(s[0][2:])
        numbers.append((I,i))
    
    sort_num = [numbers[0]]
    for j in range(1,len(numbers)):
        
        a = numbers[j]
        k = sort_num[0]
        teller = 0
        while teller<=len(sort_num)-1 and sort_num[teller][0]<a[0]:
            teller += 1
        if teller ==0:
            sort_num = [a] + sort_num[teller:]
        else:
            sort_num = sort_num[:teller] + [a] + sort_num[teller:]
        #print sort_num,teller
    #print sort_num
    
    for i in range(len(l)):
        sort.append(l[sort_num[i][1]])
    #print numbers
    #print sort
    return sort
    
if __name__ == "__main__":

    table = {'N=100 (iter,gamma,err)'  : [],
             'N=200 (iter,gamma,err)'  : [],
             'N=800 (iter,gamma,err)'  : [],
             'N=1000 (iter,gamma,err)' : [],
             'N=2000 (iter,gamma,err)' : [],}


    table['N=100 (iter,gamma,err)'].append(1)
    table['N=200 (iter,gamma,err)'].append(2)
    table['N=800 (iter,gamma,err)'].append(3)
    table['N=1000 (iter,gamma,err)'].append(4)
    table['N=2000 (iter,gamma,err)'].append(5)

    data = pd.DataFrame(table)
    #print data
    #print sorted(data.columns)
    #print NequalNum_sort(data.columns)

    data2=data.reindex_axis(NequalNum_sort(data.columns), axis=1)
    print data2
