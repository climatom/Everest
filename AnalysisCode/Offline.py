#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os, sys, itertools

# Get dir name of parent directory
parent=os.path.dirname(os.getcwd()); sys.path.insert(0, parent); datadir=parent+"/Data/"

import GeneralFunctions as GF # Note that we add to the path variable

# File name
fname=datadir+"AllStations.xlsx"

# Places
places=["Pheriche","Kala_Patthar","Pyramid","SCol"]
obs={}

# Read in all observational data to Pandas DataFrame
for ii in range(len(places)): obs[places[ii]]=\
    pd.read_excel(fname,sheet_name=places[ii],index_col=0,parse_dates=True)
    

#subset={}
#test=obs["Kala_Patthar"].apply(pd.to_numeric)
#
#    df.apply(pd.to_numeric) 
## Subset data: only take during T <0
