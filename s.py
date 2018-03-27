import pickle
import numpy as np
with open('train','rb') as mysavedata:
    data = pickle.load(mysavedata)
with open('test','rb') as mysavedata:
    test = pickle.load(mysavedata)


label = data[:,-1]
label=np.reshape(label,[-1,1])
data = data[:,:11]

labely = test[:,-1]
labely=np.reshape(labely,[-1,1])
datay = test[:,:11]






def pred(logic,label):
    pre = logic
    lab=label
    c=[]
    for i in range(len(test)):
        d=(pre[i]+0.0000000000001)/(lab[i]+0.0000000000001)
        c.append(d)
    acc = tf.reduce_mean(c)
    return pre, acc

