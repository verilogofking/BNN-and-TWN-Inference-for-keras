# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:45:09 2018

@author: VLSI-AIR
"""

from __future__ import print_function

import h5py


# 開啟檔案
filename = 'CNN132.txt'
fp = open(filename, 'w')



weight_file_path = 'CNN131.h5'

"""
BNN:
Prints out & Write to .txt file the structure of HDF5 file.

Args:
weight_file_path (str) : Path to the file to analyze
filename (str) : write into this file
"""

f = h5py.File(weight_file_path)
try:
    if len(f.attrs.items()):
        print("{} contains: \n".format(weight_file_path))
        fp.writelines("{} contains: \n".format(weight_file_path))
        
        print("Root attributes:\n")
        fp.writelines("Root attributes:\n")
        
    for key, value in f.attrs.items():
        print("  {}: {}".format(key, value))
        fp.writelines("  {}: {}\n".format(key, value))
        #fp.writelines(key)
        #fp.writelines(value)

   
    for layer, g in f.items():
        print("  {}".format(layer))
        fp.writelines("  {}\n".format(layer))
        print("    Attributes:")
        fp.writelines("    Attributes:\n")
        for key, value in g.attrs.items():
            print("      {}: {}".format(key, value))
            fp.writelines("      {}: {}\n".format(key, value))
            
        print("    Dataset:")
        fp.writelines("    Dataset:\n")
        for p_name in g.keys():
            param = g[p_name]
            subkeys = param.keys()
            for k_name in param.keys():
                #print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
                #fp.writelines("      {}/{}: {}\n".format(p_name, k_name, param.get(k_name)[:]))
                fp.writelines("      {}/{}:\n".format(p_name, k_name))
                '''
                fp.writelines("[ ")
                for i in  range(len(param.get(k_name))):
                    fp.writelines("[")
                    for j in range(len(param.get(k_name)[:][i])):
                        fp.writelines(" ",str(param.get(k_name)[i][j])," ")
                    fp.write(" ]")
                fp.writelines(" ]")
                '''        
                print('shape = ', param.get(k_name).shape)
                fp.write('shape = '+ str(param.get(k_name).shape)+'\n')
                
                if(param.get(k_name).ndim == 1):
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
                    fp.writelines("      {}/{}: {}\n".format(p_name, k_name, param.get(k_name)[:]))
                elif(param.get(k_name).ndim == 2):
                    print("[\n ",end="")
                    fp.writelines("[\n ")
                    for i in range(len(param.get(k_name))):
                        print("[ ",end="")
                        fp.writelines("[ ")
                        for j in range(len(param.get(k_name)[i])):
                            print(str(param.get(k_name)[i][j])," ",end="")
                            fp.writelines(" "+str(param.get(k_name)[i][j])+" ")
                        print("]\n",end="")
                        fp.writelines(" ]\n")
                    print("\n]")
                    fp.writelines(" ]\n")
                elif(param.get(k_name)[:].ndim == 4):
                    print("[\n ",end="")
                    fp.writelines("[\n ")
                    for i in range(len(param.get(k_name)[:])):
                        print("[ ",end="")
                        fp.writelines("[ ")
                        for j in range(len(param.get(k_name)[:][i])):
                            print("[ ",end="")
                            fp.writelines("[ ")
                            for m in range(len(param.get(k_name)[:][i][j])):
                                print("[ ",end="")
                                fp.writelines("[ ")
                                for n in range(len(param.get(k_name)[:][i][j][m])):
                                    print(str(param.get(k_name)[i][j][m][n])," ",end="")
                                    fp.writelines(" "+str(param.get(k_name)[i][j][m][n])+" ")
                                print("]\n",end="")
                                fp.writelines(" ]\n")
                            print("]\n",end="")        
                            fp.writelines(" ]\n")
                        print("]\n",end="")
                        fp.writelines(" ]\n")
                    print("\n]\n")
                    fp.writelines("\n]\n")



finally:
    f.close()
        
# 關閉檔案    
fp.close()        
        