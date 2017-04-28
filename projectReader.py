#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Katie Roberts
Machine Learning
Project - Data Reader
3.1.17
"""


import pandas as pd

#reading in data
data = pd.read_csv('Documents/sarcasmData.csv')

"""

***SIZE OF DATA***:
    
data.shape
(4692, 5)

***FIRST FIVE ITEMS IN DATA***:
    
data.head()

  Corpus Label             ID  \
0    GEN  sarc  GEN_sarc_0000   
1    GEN  sarc  GEN_sarc_0001   
2    GEN  sarc  GEN_sarc_0002   
3    GEN  sarc  GEN_sarc_0003   
4    GEN  sarc  GEN_sarc_0004   

                                          Quote Text  \
0  First off, That's grade A USDA approved Libera...   
1  watch it. Now you're using my lines. Poet has ...   
2  Because it will encourage teens to engage in r...   
3  Obviously you missed the point. So sorry the t...   
4  This is pure paranoia. What evidence do you ha...   

                                       Response Text  
0  Therefore you accept that the Republican party...  
1  More chattering from the peanut gallery? Haven...  
2  Yep, suppressing natural behavior is always th...  
3  I guess we all missed your point Justine, what...  
4  Evidence, I dont need no sticking evidence. Th...  

"""