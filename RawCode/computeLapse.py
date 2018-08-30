#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:56:01 2017

@author: tom
"""
from pylab import*
import os, numpy as np 
from netCDF4 import Dataset
import netCDF4
from numba import jit
import GeneralFunctions as GF

#=============================================================================#
# Functions
#=============================================================================#
def wrap(cmd):
    fail=os.system(cmd)
    assert fail==0,"\n\n******\nThis call failed:\n%s\n******\n\n"%cmd


def lapse_rate(fin,zname,pname,tname,high_p,low_p):

    """
    Take variables in fin -- which is time/plevel netCDF4 Dataset and 
    compute the lapse rate between high_p and low_p
    
    fin:   -> netCDF4 input file
    zname: -> name of geopotential height variable
    pname: -> name of pressure level variable
    tname: -> name of temperature level variable 
    z_c:   -> height to interpolate to
    high_p:-> height of HIGHER pressure surface (LOWER elevation) 
    low_p: -> height of LOWER pressure surface (HIGHER elevation) 


    """
    # Assignments and preallocation
    time=fin.variables["time"] # Time - needed for writing human-readable text
    p=np.squeeze(fin.variables[pname][:]) # Pressure level 
    z_high=np.squeeze(fin.variables[zname][:,p==high_p]) # Height of HIGHER pressure surface (LOWER elevation) 
    z_low=np.squeeze(fin.variables[zname][:,p==low_p]) # Height of LOWER pressure surface (LOWER elevation) 
    t_high=np.squeeze(fin.variables[tname][:,p==high_p]) 
    t_low=np.squeeze(fin.variables[tname][:,p==low_p])
    ptime=netCDF4.num2date(time[:],units=time.units,calendar=time.calendar) 
    year,mon,day,hour,pyTimes=GF.conTimes(time_str=time.units,calendar=time.calendar,times=time[:],safe=False) 
    out=np.zeros((len(year),5))
    out[:,0]=year; out[:,1]=mon; out[:,2]=day; out[:,3]=hour
    # Compute lapse
    out[:,4]=(t_high-t_low)/((z_high-z_low)/9.80665)*1000.0 # to give degC (or K) per 1000 m
            
    return out

#=============================================================================#
# User-defined args
base="/media/tom/WD12TB/Everest/ERA-I_" # folder with lapse rate data in...
# Everest coords
lonlat="lon=86.9250_lat=27.9878"
header="year\tmonth\tday\thour\tLapse"
#=============================================================================#
count=0
for yr in range(1979,2017):
    for mon in range(1,13):
        
        fname=base+"%02d_%02d.nc"%(mon,yr)
        
        # Use cdo to interpolate horizontally
        cmd="cdo -L -s -b 32 remapbil,%s %s scratch.nc"%(lonlat,fname)
        wrap(cmd)

        # Open file
        fin=Dataset("scratch.nc","r")
        value=lapse_rate(fin,"z","level","t",400,300)
        
        if count==0:
            out=value*1.
        else:
            out=np.vstack((out,value))
    
        count+=1; print "Finished computing the lapse rate from file: %.0f" % count
        
np.savetxt("Interpolated_Lapse.txt",out,fmt="%.3f",header=header)

