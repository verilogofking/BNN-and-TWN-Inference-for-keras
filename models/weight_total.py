# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 18:15:52 2018

@author: VLSI-AIR
"""

from __future__ import print_function
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  

weight_file_path = 'LSTM0729_twn5.h5'

total = []

"""
Prints out the structure of HDF5 file.

Args:
weight_file_path (str) : Path to the file to analyze
"""
f = h5py.File(weight_file_path)
try:
    if len(f.attrs.items()):
        print("{} contains: ".format(weight_file_path))
        print("Root attributes:")
    for key, value in f.items():
        print("  {}: {}".format(key, value))
    for key, value in f.attrs.items():
        print("  {}: {}".format(key, value))

   
    for layer, g in f.items():
        print("  {}".format(layer))
        print("    Attributes:")
        for key, value in g.attrs.items():
            print("      {}: {}".format(key, value))

        print("    Dataset:")
        for p_name in g.keys():
            param = g[p_name]
            subkeys = param.keys()
            for k_name in param.keys():
                print("      {}/{}:\n \tshape = {} \n{}".format(p_name, k_name,  param.get(k_name)[:].shape, param.get(k_name)[:]))
                total.append(param.get(k_name)[:])#
                
    #print(f['dense_1']['dense_1']['bias:0'][0]) # read 1 element
    #for key in f['dense_1']['dense_1']['bias:0']:
       # print(key)

finally:
    f.close()
    
############# Flatten ##########################################   
weight_flatten = []
Total_weight_flatten = []

for i,array in enumerate(total):
    if array.ndim == 1 :
        weight_flatten.append(array)
    elif array.ndim == 2:
        weight_flatten.append(total[i].flatten())
    elif array.ndim == 4:
        weight_flatten.append(total[i].reshape(-1))
    else:
        print('error')
Total_weight_flatten = [y for x in weight_flatten for y in x]      
################################################################

#中位數
print('Median = ',np.median(Total_weight_flatten))
#四分位距
print('Q1,Q2,Q3 = ',np.percentile(Total_weight_flatten, [25, 50, 75]))

#最大最小值
print('Max = ',np.max(Total_weight_flatten))
print('Min = ',np.min(Total_weight_flatten))

############# Plot #########################################
plt.style.use('ggplot')

num_bins = 200
n, bins, patches = plt.hist(Total_weight_flatten, num_bins, density=0, facecolor='g', alpha=0.75)

plt.xlabel('Value of weights',{'size': 30})
plt.ylabel('Count',{'size': 30})
plt.title('Histogram of weights',{'size': 30})
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([0.5, -0.5,0,5000])
plt.grid(True)
#刻度字大小
plt.tick_params(labelsize=23)

plt.show()

'''
############# Plot with Line #########################################
num_bins = 50  
# the histogram of the data  
n, bins, patches = plt.hist(Total_weight_flatten, num_bins, density=True, facecolor='blue', alpha=0.5)  
# add a 'best fit' line  
y = mlab.normpdf(bins, np.array(Total_weight_flatten).mean(), np.array(Total_weight_flatten).std())  
plt.plot(bins, y, 'r--')  
plt.xlabel('Smarts')  
plt.ylabel('Probability')  
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')  
      
# Tweak spacing to prevent clipping of ylabel  
#plt.subplots_adjust(left=0.15)  
plt.show()  
'''