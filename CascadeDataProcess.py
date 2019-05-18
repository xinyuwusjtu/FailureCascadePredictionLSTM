import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd

## Try to save a list variable in txt file.
def text_save(content,filename,mode='a'):
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()

## Try to read a txt file and return a list.Return [] if there was a mistake.
def text_read(filename):
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]

    file.close()
    return content

######################################## Load data ##################################################

##Load the sampled cascade sequence
link_size=41
a=np.ones([0,link_size])
for j in range(10):
    print(j)
    workbook = xlrd.open_workbook("C:\\Users\\wuxinyu\\Documents\\MATLAB\\Modiano\\ItalianGrid\\ItalianGrid\\ItalianGrid\\history_link_ieee30_rnd_2to11_"+str(j+1)+".xlsx")
    booksheet = workbook.sheet_by_index(0)       
    for i in range(booksheet.ncols):
        a=np.vstack((a,booksheet.col_values(i)))

workbook=xlrd.open_workbook("C:\\Users\\xinyuwu1\\Documents\\MATLAB\\ItalianGrid\\History_link_ieee30_4_FixedCapacity_0_8_AC.xlsx")
booksheet = workbook.sheet_by_index(0)
for i in range(booksheet.nrows):
    print(i)
    a=np.vstack((a,booksheet.row_values(i)))

np.save("a_30_FailSize_4_FixedCapacity_0_8_AC.npy",a)
##data=np.load("a_30_FailSize_4_FixedCapacity_0_8_AC.npy")

######################################## Deal with data ##################################################

cascade=np.load("a_30_FailSize_4_FixedCapacity_0_8_AC.npy")
cascade=cascade.T
max_data=cascade.max()
M=len(cascade[:,1]) #M: number of links
K=len(cascade[1,:]) #K: number of samples

state_series_tensor=np.zeros((M,int(max_data),K))
size_state_vector=np.zeros(K)
cascade=cascade.astype(np.int)

for i in range(K):
    if i % 1000 ==0:
        print(i)
    tmp_cascade=np.zeros(M)
    for j in range(M):
        tmp_cascade[j]=cascade[j,i]
    max_i=tmp_cascade.max()
    for j in range(M):
        if cascade[j,i]==max_i:
            tmp_cascade[j]=0
    submax_i=tmp_cascade.max()
    if max_i-submax_i>=2:
        size_state=submax_i+1 
    else:
        size_state=submax_i+1
    state_series=np.ones((M,int(size_state)))
    state_series=state_series.astype(np.int)

    for t in range(int(size_state)):
        for k in range(M):
            if cascade[k,i] <= t+1:
                state_series[k,t]=0
                
    state_series_tensor[:,0:int(size_state),i]=state_series;
    for j in range(int(max_data)-int(size_state)):
        state_series_tensor[:,int(size_state)+j,i]=state_series[:,int(size_state)-1]
    size_state_vector[i]=int(size_state);

##state_series_tensor: M*T*K tensor. M links, K samples, T denotes the max time length of cascade among all K samples
##size_state_vector: K*1 vector. The k-th element represents the time length of the k-th cascade sequence.
np.save("SST_30_FailSize_4_FixedCapacity_0_8_AC.npy",state_series_tensor)
np.save("SSV_30_FailSize_4_FixedCapacity_0_8_AC.npy",size_state_vector)


