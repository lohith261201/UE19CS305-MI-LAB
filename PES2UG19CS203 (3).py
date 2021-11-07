'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float


def get_entropy_of_dataset(df):
    target = df[[df.columns[-1]]].values
    _, counts = np.unique(target, return_counts=True)
    total_count = np.sum(counts)
    entropy = 0
    for freq in counts:
        entropy = entrval(entropy,total_count, freq)
    return entropy

def entrval(a,total_count, freq):
    dummy=freq/total_count
    temp = dummy
    if dummy != 0:
        a = a - dummy*(np.log2(dummy))
    return a


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float


def get_avg_info_of_attribute(df, attribute):
    entropy_of_attribute = 0
    attribute_values = df[attribute].values
    unique_attribute_values = np.unique(attribute_values)
    rows = df.shape[0]
    for current_value in unique_attribute_values:
        entropy_of_attribute = entrattribfind(df, rows, attribute, current_value, entropy_of_attribute)
    return(abs(entropy_of_attribute))

def entrattribfind(df, rows, attribute, current_value, entropy_of_attribute):
    entropy = 0
    df_slice = df[df[attribute] == current_value]
    temp = df_slice[[df_slice.columns[-1]]].values
    _, counts = np.unique(temp, return_counts=True)
    numb = np.sum(counts)
    for freq in counts:
        entropy = entropval(numb, entropy, freq)
    entropy_of_attribute += entropy*(np.sum(counts)/rows)
    return entropy_of_attribute

def entropval(num, n ,freq):
    temp = freq/num
    dummy = temp
    if temp != 0:
        n -= temp*np.log2(temp)
    return n


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float


def get_information_gain(df, attribute):
    entrof_dataset = get_entropy_of_dataset(df)
    entrof_attr = get_avg_info_of_attribute(df, attribute)
    info_gain = 0
    fake = entrof_dataset - entrof_attr
    info_gain= fake
    return info_gain


# input: pandas_dataframe
# output: ({dict},'str')
def get_selected_attribute(df):
    info_gains = {}

    '''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected
	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''
    selectedcol = ''
    maxinfogain = float("-inf")
    for attr in df.columns[:-1]:
        infogainofattr = get_information_gain(df, attr)
        if infogainofattr > maxinfogain:
            maxinfogain = infogainofattr
            selectedcol = attr
        info_gains[attr] = infogainofattr
    return (info_gains, selectedcol)
