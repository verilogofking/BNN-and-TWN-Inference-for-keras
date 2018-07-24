from __future__ import print_function

import h5py



weight_file_path = 'LSTM0724_twn3.h5'

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

    #print(f['dense_1']['dense_1']['bias:0'][0]) # read 1 element
    #gg=f['lstm_1']['lstm_1']['recurrent_kernel:0'][0][0][0][0]
    #print('=====\n',f['lstm_1']['lstm_1']['recurrent_kernel:0'][0]) # read 1 element
    #for key in f['dense_1']['dense_1']['bias:0']:
       # print(key)

finally:
    f.close()
 

#print(f.attrs.items())       
#print(f['dense_1'])

        
        
'''
def print_structure(weight_file_path):
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
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

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
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()

'''