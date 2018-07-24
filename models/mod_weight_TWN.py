# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:58:03 2018

@author: VLSI-AIR
"""



from __future__ import print_function

import h5py

import numpy as np

weight_file_path = 'LSTM0729_twn5.h5'

PTHRESHOLD = 0.02524744
NTHRESHOLD = -0.02435329
PVALUE = 0.25
NVALUE = -0.25
ZVALUE = 0
"""
###
TWN:
    w = weight    
    if(w>0.5) w=1 ;if(0.5<=w>=-0.5) w=0 ;if(w<-0.5) w=-1    
    +0.5 -> PTHRESHOLD = positive threshold
    -0.5 -> NTHRESHOLD = negative threshold
    +1   -> PVALUE = positive value
    -1   -> NVALUE = negative value
     0   -> ZVALUE = bias value
###

Prints out the structure of HDF5 file.

Args:
    weight_file_path (str) : Path to the file to analyze
    
##  param[k_name] = param.get(k_name)
    param[k_name][0] to modify one element for 1D array 
    param[k_name][0][0] to modify one element for 2D array 
    f['layer_name']['layer_name']['bias:0'/'kernel:0'][1D][2D][3D][4D]
"""
f = h5py.File(weight_file_path)
try:
    if len(f.attrs.items()):
        print("{} contains: ".format(weight_file_path))
        print("Root attributes:")
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
                #print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
                print("ha--------------------------------")
                print('dim = ', param.get(k_name).ndim)
                
                # dimension 1 - bias
                if(param.get(k_name)[:].ndim == 1):
                    
                    for i in range(len(param.get(k_name)[:])):
                        if  ( param.get(k_name)[i] < NTHRESHOLD ): #negetive
                            param.get(k_name)[i] = NVALUE
                        elif( param.get(k_name)[i] > PTHRESHOLD ): #positive
                            param.get(k_name)[i] = PVALUE               
                        else:
                            param.get(k_name)[i] = ZVALUE
                # dimension 2 - LSTM . FC    
                elif(param.get(k_name)[:].ndim == 2):
                   
                    for i in range(len(param.get(k_name)[:])):
                        array_1D = np.zeros(len(param.get(k_name)[i]))
                        for j in range(len(param.get(k_name)[:][i])):
                            if  ( param.get(k_name)[i][j] < NTHRESHOLD ):
                                array_1D[j] = NVALUE
                            elif( param.get(k_name)[i][j] > PTHRESHOLD ):
                                array_1D[j] = PVALUE
                            else:
                                array_1D[j] = ZVALUE
                                
                        param.get(k_name)[i] = array_1D
                # dimension 4 - CNN
                elif(param.get(k_name)[:].ndim == 4):
                    for i in range(len(param.get(k_name)[:])):
                        array_3D = np.zeros(param.get(k_name)[:][i].shape)
                        for j in range(len(param.get(k_name)[:][i])):
                            for m in range(len(param.get(k_name)[:][i][j])):
                                for n in range(len(param.get(k_name)[:][i][j][m])):
                                    if( param.get(k_name)[i][j][m][n] < NTHRESHOLD ):
                                        array_3D[j][m][n] = NVALUE
                                    elif( param.get(k_name)[i][j][m][n] > PTHRESHOLD ):
                                        array_3D[j][m][n] = PVALUE
                                    else:
                                        array_3D[j][m][n] = ZVALUE

                        param.get(k_name)[i] = array_3D
                                        
                print("      {}/{}:\n {}".format(p_name, k_name, param.get(k_name)[:]))

finally:
    f.close()
    
