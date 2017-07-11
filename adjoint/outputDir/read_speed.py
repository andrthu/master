import sys
import pandas as pd
import numpy as np

def read_file(filename):

    speed_file = open(filename,'r')

    lines = speed_file.readlines()

    speed_file.close()

    return lines
def create_table(lines,with_iter=True,with_iter2=False,with_norm=False,ideal_S=False):
    
    if with_iter:
        table = {'time':[],'iter':[],'ls iter':[],'speedup':[]}
    elif with_iter2:
        table = {'time':[],'iter':[],'ls iter':[],'speedup':[],'gr':[],'fu':[]}
    elif with_norm:
        table = {'time':[],'iter':[],'ls iter':[],'speedup':[],'gr':[],'fu':[],'norm':[]}
        with_iter2 =True
    elif ideal_S:
        table = {'time':[],'L':[],'speedup':[],'S2':[],'norm':[],'f':[],'E':[]}
    else:
        table = {'time':[],'speedup':[]}
    index = []
    
    first_line = lines[0].split()
    #print first_line
    index.append(1)
    seq_time =float(first_line[1])
    seq_fugr =1
    table['time'].append(float(first_line[1]))
    if with_iter:
        table['iter'].append(int(first_line[2]))
        table['ls iter'].append(int(first_line[3]))
    elif with_iter2:
        table['iter'].append(int(first_line[4]))
        table['ls iter'].append(int(first_line[5]))
        table['gr'].append(int(first_line[3]))
        table['fu'].append(int(first_line[2]))
        if with_norm:
            table['norm'].append('--')
    elif ideal_S:
        table['L'].append(int(first_line[3])+int(first_line[2]))
        table['S2'].append(1)
        
        table['f'].append('--')
        table['E'].append(1)
        table['norm'].append(float(first_line[-1]))
        seq_fugr = float(int(first_line[3])+int(first_line[2]))
    table['speedup'].append(1)
    
    for line in lines[1:]:
        line_list = line.split()
        #print line_list 
        index.append(int(line_list[1][:-1]))
        par_time = float(line_list[2])
        table['time'].append(par_time)
        if with_iter:
            table['iter'].append(int(line_list[3]))
            table['ls iter'].append(int(line_list[4]))
        elif with_iter2:
            table['iter'].append(int(line_list[5]))
            table['ls iter'].append(int(line_list[6]))
            table['gr'].append(int(line_list[4]))
            table['fu'].append(int(line_list[3]))
            if with_norm:
                table['norm'].append(float(line_list[7]))
        elif ideal_S:
            table['L'].append(int(line_list[4])+int(line_list[3]))
            table['S2'].append(int(line_list[1][:-1])*seq_fugr/(int(line_list[4])+int(line_list[3])))
            table['norm'].append(float(line_list[7]))
            table['f'].append(float(line_list[9]))
            
        table['speedup'].append(seq_time/par_time)
        if ideal_S:
            table['E'].append(seq_time/(par_time*int(line_list[1][:-1])))
    #print table
    data = pd.DataFrame(table,index=index)
    if ideal_S:
        data = data.ix[:,['norm','f','L','S2','time','speedup','E']]
    return data
    
def main():
    ideal_S = False
    try:
        if sys.argv[1] == '0':
            names = sys.argv[2:]
            with_iter = False
            with_iter2 = False
            with_norm =False
            ideal_S=False
        elif sys.argv[1] == '1':
            names = sys.argv[2:]
            with_iter = False
            with_iter2 = True
            with_norm = False
            ideal_S=False
        elif sys.argv[1] == '2':
            names = sys.argv[2:]
            with_iter = False
            with_iter2 = False
            with_norm =True
            ideal_S=False
        elif sys.argv[1] == '3':
            names = sys.argv[2:]
            with_iter = False
            with_iter2 = False
            with_norm =False
            ideal_S=True
        else:
            names = sys.argv[1:]
            with_iter = True
            with_iter2 = False
            with_norm =False
            ideal_S=False
    except:
        print 'Give file'
        return
    datas = []
    for name in names:
        print name
        lines = read_file(name)
        data = create_table(lines,with_iter=with_iter,with_iter2=with_iter2,
                            with_norm=with_norm,ideal_S=ideal_S)
        print data
        #data.to_latex('latexTables/'+name[:-3]+'tex')
        datas.append(data)

    if ideal_S:
        make_a_plot2(datas)

def make_a_plot(datas):

    import matplotlib.pyplot as plt
    legg = [r'$n=6\cdot 10^{5}$',r'$n=12\cdot 10^{5}$',r'$n=24\cdot 10^{5}$']
    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(221)
    ax1.set_ylabel('speedup')

    N_vals = [1,4,8,16,24,32,40,48,56,64,72,80,88,96,104,120]
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['speedup']
        ax1.plot(N[:-2],datas[i]['speedup'][:-2],'o-')
        
    ax1.xaxis.set_ticks(N_vals)
    ax1.legend(legg,loc='best',fontsize='medium')
    ax1.set_xlabel(r'$N$')

    ax2 = plt.subplot(222)
    ax2.set_ylabel('Efficiency')
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['E']
        ax2.plot(N[:-2],datas[i]['E'][:-2],'o-')
    ax3 = plt.subplot(224)
    ax3.set_ylabel(r'$L_{p_N}$',fontsize=15)
    ax2.xaxis.set_ticks(N_vals)
    ax2.legend(legg,loc='best',fontsize='medium')
    ax2.set_xlabel(r'$N$',ha='left',va='top')
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['E']
        ax3.plot(N[:-2],datas[i]['L'][:-2],'o-')
    ax3.xaxis.set_ticks(N_vals)
    ax4 = plt.subplot(223)
    ax4.set_ylabel(r'$||v-v_e||_{L^2}$',fontsize=15)
    ax3.legend(legg,loc='best',fontsize='medium')
    ax3.set_xlabel(r'$N$')
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['norm']
        ax4.plot(N[:-2],datas[i]['norm'][:-2],'o-')
    ax4.xaxis.set_ticks(N_vals)
    ax4.legend(legg,loc='best',fontsize='medium')
    ax4.set_xlabel(r'$N$')
    

    print datas[-1].iloc[[0,1,3,5,7,10,13,16]]
    #datas[-1].iloc[[0,1,3,5,7,10,13,16]].to_latex('latexTables/speedupManyCore.tex')
    plt.savefig('report/draft/draft2/fspeed_fig.png')
    plt.show()

def make_a_plot2(datas):

    import matplotlib.pyplot as plt
    legg = [r'$n=6\cdot 10^{5}$',r'$n=12\cdot 10^{5}$',r'$n=24\cdot 10^{5}$']
    legg2 = [r'$n=6\cdot 10^{5}\ S$',r'$n=6\cdot 10^{5}\ \hat S$',r'$n=12\cdot 10^{5}\ S$',r'$n=12\cdot 10^{5}\ \hat S$',r'$n=24\cdot 10^{5}\ S$',r'$n=24\cdot 10^{5}\ \hat S$']
    plt.figure(figsize=(12,6))
    
    

    N_vals = [1,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120]
    N_vals2 = [2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120]
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['speedup']
        plt.plot(N[:-2],datas[i]['speedup'][:-2],'o-')
        #plt.plot(N[:-2],datas[i]['S2'][:-2],'v-')
    plt.ylabel('speedup',fontsize=20)    
    plt.xticks(N_vals)
    plt.legend(legg,loc='best',fontsize=20)
    plt.xlabel(r'$N$',fontsize=20)
    plt.yticks(fontsize=15)
    #plt.savefig('presentation/SS.png')
    plt.show()
    
    plt.figure(figsize=(12,6))
    for i in range(len(datas)):
        N= datas[i].index
        val =datas[i]['speedup'][:-2]/datas[i]['S2'][:-2]
        plt.plot(N[:-2],val,'o-')
    plt.xticks(N_vals2)
    plt.legend(legg,loc='best',fontsize=20)
    plt.xlabel(r'$N$',fontsize=20)
    plt.ylabel(r'$\frac{S}{\hat S} \quad $',fontsize=30,rotation=0)
    plt.yticks(fontsize=15)
    #plt.savefig('presentation/SS2.png')
    plt.show()
    print datas[i]['S2']
    print datas[i]['speedup']
    plt.figure(figsize=(12,6))
    
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['norm']
        plt.plot(N[:-2],datas[i]['norm'][:-2],'o-')
    plt.xticks(N_vals)
    plt.legend(legg,loc='best',fontsize=20)
    plt.xlabel(r'$N$',fontsize=20)
    plt.ylabel(r'$||v-v_e||_{L^2}$',fontsize=20)
    plt.show()

    plt.figure(figsize=(12,6))

    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['E']
        plt.plot(N[:-2],datas[i]['L'][:-2],'o-')
    plt.xticks(N_vals)
    plt.legend(legg,loc='best',fontsize=20)
    plt.xlabel(r'$N$',fontsize=20)
    plt.ylabel(r'$L_{p_N}$',fontsize=20,rotation=0)
    plt.show()
    return
    plt.figure(figsize=(12,6))
    ax2 = plt.subplot(222)
    ax2.set_ylabel('Efficiency')
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['E']
        ax2.plot(N[:-2],datas[i]['E'][:-2],'o-')
    ax3 = plt.subplot(224)
    ax3.set_ylabel(r'$L_{p_N}$',fontsize=15)
    ax2.xaxis.set_ticks(N_vals)
    ax2.legend(legg,loc='best',fontsize='medium')
    ax2.set_xlabel(r'$N$',ha='left',va='top')
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['E']
        ax3.plot(N[:-2],datas[i]['L'][:-2],'o-')
    ax3.xaxis.set_ticks(N_vals)
    ax3.legend(legg,loc='best',fontsize='medium')
    ax3.set_xlabel(r'$N$')
    ax4 = plt.subplot(223)
    ax4.set_ylabel(r'$||v-v_e||_{L^2}$',fontsize=15)
    
    for i in range(len(datas)):
        
        N= datas[i].index
        #print N
        #print datas[i]['norm']
        ax4.plot(N[:-2],datas[i]['norm'][:-2],'o-')
    ax4.xaxis.set_ticks(N_vals)
    ax4.legend(legg,loc='best',fontsize='medium')
    ax4.set_xlabel(r'$N$')
    

    print datas[-1].iloc[[0,1,3,5,7,10,13,16]]
    #datas[-1].iloc[[0,1,3,5,7,10,13,16]].to_latex('latexTables/speedupManyCore.tex')
    plt.savefig('report/draft/draft2/fspeed_fig.png')
    plt.show()


    

def read_func_and_grad():

    names = sys.argv[1:]
    
    n = len(names)
    
    names1 = names[:n/2]
    names2 = names[n/2:]
    #print names1
    #print
    #print names2

    for i in range(len(names1)):
        table = {'functinal time(s)':[],
                 'gradient time(s)':[],
                 'functional speedup':[],
                 'gradient speedup':[], }

        func_lines = read_file(names1[i])
        grad_lines = read_file(names2[i])
        
        first_func = func_lines[0].split()
        func_seq = float(first_func[1])
        table['functinal time(s)'].append(func_seq)
        table['functional speedup'].append(1)
        
        first_grad = grad_lines[0].split()
        grad_seq = float(first_grad[1])
        table['gradient time(s)'].append(grad_seq)
        table['gradient speedup'].append(1)
        
        index = [1]
        
        for j in range(1,len(func_lines)):
            func_line = func_lines[j].split()
            func_time = float(func_line[2])
            table['functinal time(s)'].append(func_time)
            table['functional speedup'].append(func_seq/func_time)
            
            grad_line = grad_lines[j].split()
            grad_time = float(grad_line[2])
            table['gradient time(s)'].append(grad_time)
            table['gradient speedup'].append(grad_seq/grad_time)
            index.append(j+1)

        data = pd.DataFrame(table,index=index)
        
        data=data.ix[:,['functinal time(s)','gradient time(s)','functional speedup','gradient speedup']]
        print names1[i].split('/')[-1][:-4]
        print data
        
        help_name = names1[i].split('/')[-1][:-4]
        #data.to_latex('latexTables/grad'+help_name+'.tex')
        #print names1[i],names2[i]

if __name__ == '__main__':
    main()
    #read_func_and_grad()
