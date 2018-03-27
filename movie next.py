import pandas as pd
import numpy as np
import pickle
with open('ccc','rb') as mysavedata:
    cc = pickle.load(mysavedata)

nameset= pd.read_excel('nameset.xlsx',)
nameset=np.array(nameset)
name = pd.read_excel('name.xlsx')
name = np.array(name)
name1 = name[:,0]
name2 = name[:,1]
name3 = name[:,2]
c=[]
f=[]

for i in range(len(nameset)):

    for j in range(len(name1)):
        if name3[j] == i:
            c.append(name2[j])
        else:
            continue
    f.append(c)
    c=[]

print(f[11])

mean=[]
for i in range(len(f)):
    bb=np.mean(f[i],dtype='float16')
    mean.append(bb)

mean1=[]


for i in range(len(nameset)):

    for j in range(len(name1)):
        if name3[j] == i:
            mean1.append(mean[i])
        else:
            continue


num=[]
for i in range(len(f)):
    a=len(f[i])
    num.append(a)


num1=[]
for i in range(len(nameset)):

    for j in range(len(name1)):
        if name3[j] == i:
            num1.append(num[i])
        else:
            continue

print(num1)

index_format = mean1
columns_format = ['x', 'y']

# DataFrame 초기화
values = pd.DataFrame(index=index_format, columns=columns_format)

# x & y 값 정의
x = index_format
y = num1

for ii in range(values.shape[0]):
    # fill in x values into column index zero of values
    values.iloc[ii, 0] = x[ii]
    # fill in x values into column index one of values
    values.iloc[ii, 1] = y[ii]

# saves DataFrame(values) into an Excel file
values.to_excel('./test.xlsx',
                sheet_name='Sheet1',
                columns=columns_format,
                header=True,
                index=index_format,
                index_label="y = sin(x)",
                startrow=1,
                startcol=0,
                engine=None,
                merge_cells=True,
                encoding=None,
                inf_rep='inf',
                verbose=True,
                freeze_panes=None)
