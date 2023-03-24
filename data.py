import meshing
import os
import time
import pandas as pd
import numpy as np

directory='N:/54_QuaSAR/DATA_CENTER/Testdatensatz_Enovia/models/'
#directory='C:/ManSWInst/D/CODING/PYTHON_CODING/Neu/CAE_Project'
i=0
l=[]
names=['M200','M020','M002','M022','M202','M220','M300','M030','M003','M033','M303','M330','M400','M040','M004','weight','Type']
dfMoments=pd.DataFrame([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],index=names,columns=['Filename'])
i=0
for root, dirs, files in os.walk(directory, topdown=False):
    try:
        for file in files:
        
            if '.stp' in file:
                
                try:
                    file='{}/{}'.format(root.replace('\\','/'),file)
                                    
                    caType, weight, nodeTagsInner, nodeTagsOuter,feature = meshing.getABQFromStpFile(file,elmax=8,elmin=1)
                    features=np.append(feature,[weight,caType])
                    dfMoments[file]=features
                    
                    #l.append([files[0], caType, weight])
                    
                    print('{} analyzed {}'.format(time.ctime(),file))
                    i+=1
                    print(str(i)+' Files'+' analyzed')
                    
                except:

                    print('{} Failed'.format(file))
                    pass
    except:
        pass
    
print(dfMoments)
dfMoments.to_csv('momentsEnovia_1.csv')

