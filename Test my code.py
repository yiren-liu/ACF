
import pandas as pd
import numpy as np
photoid = []
featureData = []
with open('name-fc8.txt', 'r') as file:
    for line in file.readlines():
        a,b=line.split(' ', 1)
        a=eval(a)[0].split('.')[0]
        b=eval(b)
        if a in photoid:
            continue
        photoid.append(a)
        featureData.append(b)
photoid=pd.factorize(pd.Series(photoid))[1]
featureData=np.array(featureData,dtype='float32')