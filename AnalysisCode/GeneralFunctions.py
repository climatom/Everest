# -*- coding: utf-8 -*-
"""
Created on Fri May 08 16:18:16 2015
 
@author: t.matthews@lboro.ac.uk

"""
 
import numpy as np
import scipy
from scipy import signal
import datetime
import scipy.stats as stats 
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
#import statsmodels.api as sm
from netcdftime import utime
from datetime import  datetime
from netCDF4 import Dataset
from scipy.special import gammaln as gammaln
import os,sys
import nvector as nv
from random import randint
import aoslib as al
from numba import jit
import statsmodels.api as sm
#-----------------------------------------------------------------------------#
#                                Constants                                    #
#-----------------------------------------------------------------------------#
R = 6378.1370 # equatorial radius of Earth (km)
wgms84_a=6378137.0 # semi-major axis
wgms84_b=6356752.31425 # semi-minor axis
wgms84_e_sq=1-np.power((wgms84_b/wgms84_a),2)
#-----------------------------------------------------------------------------#
 
# Functions follow..

@jit
def grid(time,lon,lat,res=0.5,var=None,diff=False):
    
    """
    Simple routine that takes the lon/lat of trajectories and grids their 
    frequencies in a lat/lon grid with res=res.
    If a variable is requested (i.e. var != None), then also grid the mean value
    If 'diff' is True, then the met quantity is first differenced from the row 
    above before the mean is calculated 
    
    NOTE: 'var' should be a column vector
    """
    
    # Initialize and preallocate
    n=len(lon)
    lat0=90-res/2.; latMax=-90
    lon0=-180+res/2.; lonMax=180
    lon2,lat2=np.meshgrid(np.arange(lon0,lonMax,res),\
    np.arange(lat0,latMax,-res))
    out_freq=np.zeros((lon2.shape))
    maxt=np.min(time)
    vertlon1=np.radians(lon2-res/2.)
    vertlon2=np.radians(lon2+res/2.)
    vertlat1=np.radians(lat2-res/2.)
    vertlat2=np.radians(lat2+res/2.)
    areas=6371**2*(vertlon1-vertlon2)*(np.sin(vertlat1)-np.sin(vertlat2))

    
    # Preallocate if necessary
    if type(var) is np.ndarray:
        out_var=np.zeros(out_freq.shape)
    
    # Start looping
    for ii in range(n):
                                
        # Determine cells for output
        row=np.round((lat0-lat[ii])/res)
        col=np.abs(np.round((lon0-lon[ii])/res))
        
        # Freq
        out_freq[row,col]+=1
        
        # Do difference now
        if type(var) is np.ndarray:
            if diff:
                if time[ii]==maxt:
                    continue
                
                # Adjust the row/col to reflect the average position of air parcel
                row=row=np.round((lat0-0.5*(lat[ii]+lat[ii+1]))/res)            
                col=np.abs(np.round((lon0-0.5*(lon[ii]+lon[ii+1]))/res))
                
                # Difference 
                out_var[row,col]+=(var[ii]-var[ii+1])
                
                if (var[ii]-var[ii+1]) > 50:
                    print time[ii], ii
                    assert 1==2
                
            # Not difference - straightforward accumulation (holds 'old' row/col)
            else:
                out_var[row,col]+=var[ii]
        
    if type(var) is np.ndarray:
        out_var[out_freq!=0]/=out_freq[out_freq!=0]
        return out_var,out_freq, [lon2,lat2], areas
    else:    
        return out_freq, [lon2,lat2], areas
    
    
@jit
def distgrid(lon,lat,lat_st,lat_fin,lon_st,lon_fin,dist,res=1.): 
    
    """
    Given lon/lat points, calculate the frequency with which they pass within
    'dist' km of the coordinates given by a global grid of resolution 'res
    """    
    
    # Create grid
    lat_g=np.radians(np.arange(lat_st,lat_fin+res,res))
    lon_g=np.radians(np.arange(lon_st,lon_fin+res,res))
    # Mesh it
    lon_g,lat_g=np.meshgrid(lon_g,lat_g)

    # Constant (to get distance)
    c=2*R
    
    # Transform input coordinates to radians
    lon=np.radians(lon)
    lat=np.radians(lat)
    
    # Preallocate output grid
    out=np.zeros(lat_g.shape)
    
    # Loop over all coordinate pairs
    for ii in range(len(lat)):
               
        # define deltas
        dlon=lon_g-lon[ii]
        dlat=lat_g-lat[ii]
    
        # Intermediate calculation 
        a=np.power(np.sin(dlat/2.),2)+np.cos(lat[ii])*np.cos(lat_g)*\
        np.power(np.sin(dlon/2),2)
    
        # Distance - of shape out.shape
        d=np.multiply(np.arcsin(np.sqrt(a)),c)
        
        # Where d<=dist, +=1
        out[d<=dist]+=1
    
    return np.degrees(lon_g),np.degrees(lat_g),out

@jit
def sphericalsmooth(lons,lats,grid,thresh_km): 
    
    """
    Take a distance-weighted mean of the values in grid
    """    
    ntime,nrows,ncols=grid.shape
    rows=range(nrows); cols=range(ncols)
    lon_g,lat_g=np.meshgrid(lons[:],lats[:])
    cols_g,rows_g=np.meshgrid(cols,rows)
    rows_g=rows_g.ravel(); cols_g=cols_g.ravel()
    pos=geoCart(lat_g.ravel(),lon_g.ravel(),np.zeros(len(lon_g.ravel())))
    tree=scipy.spatial.cKDTree(pos)
    pairs=tree.query_ball_tree(tree,r=thresh_km*1000.)
    out=np.zeros(grid.shape)
    
    for tt in range(ntime):
        scratch=np.squeeze(grid[tt,:,:]).ravel()
        for ii in range(len(lat_g.ravel())):        
            if np.isnan(scratch[ii]):
                continue
            else:
                out[tt,rows_g[ii],cols_g[ii]]=np.nanmean(scratch[pairs[ii]])
                
    return out

    
def write_nc_coord2d(ofile,varis,lat_out,lon_out,x,y,time_out,\
time_string="blank",lat_string="degrees_north",\
lon_string="degrees_east",cal="standard",mv=-999.999):
    
  # note that lat_out, lon_out, time_out etc must 
  # contain full variable info (i.e. not just values)
  # varis is a dictionary of variable names: variables   
  ncfile=Dataset(ofile,"w")
  nrows=len(y); ncols=len(x); print nrows; print ncols
  try:
      ntime=len(time_out)
  except:
      ntime=1
  
  # create lat and lon dimensions
  ncfile.createDimension('time',ntime)
  ncfile.createDimension('y',nrows)
  ncfile.createDimension('x',ncols)
  
  # Define the coordinate variables. They will hold the coordinate
  # information 
  times = ncfile.createVariable('time',float,dimensions=['time',])
  lats = ncfile.createVariable('latitude',float,dimensions=['y','x'])
  lons = ncfile.createVariable('longitude',float,dimensions=['y','x',]) 
   
  try:
    times.units = time_out.units
  except:
    times.units = time_string
    
  try:
      times.calendar=time_out.calendar
  except:
      times.calendar=cal
  try:       
      lats.units = lat_out.units
  except:
      lats.units=lat_string
  try:
      lons.units=lon_out.units
  except:
      lons.units=lon_string
      
  # write data to coordinate vars.
  try:
      times[:] = time_out[:]
      
  except:
      times=time_out
      
  lats[:,:] = lat_out[:,:]
  lons[:,:] = lon_out[:,:]
  
  # now create variable(s)
  for d in varis.iteritems():
      var = ncfile.createVariable(d[0],float,dimensions=['time','latitude','longitude'],\
      fill_value = mv)
      # write data to this variable
      try:
          var[:,:,:] = d[1][:,:,:]
          var.units = d[1].units
      except:# if not netcdf, need to read explicitly

          var[:,:,:] = d[1][0]
          var.units = d[1][1]   
     
  # now close file 
  ncfile.close()   
        
 
# Function to write netcdf object(s) --> netcdf objects 
def write_nc(ofile,varis,lat_out,lon_out,time_out,\
time_string="blank",lat_string="degrees_north",\
lon_string="degrees_east",cal="standard",mv=-999.999):

  # note that lat_out, lon_out, time_out etc must 
  # contain full variable info (i.e. not just values)
  # varis is a dictionary of variable names: variables   
  ncfile=Dataset(ofile,"w")
  nrows = len(lat_out); ncols = len(lon_out)
  try:
      ntime=len(time_out)
  except:
      ntime=1
  
  # create lat and lon dimensions
  ncfile.createDimension('time',ntime)
  ncfile.createDimension('latitude',nrows)
  ncfile.createDimension('longitude',ncols)
  # Define the coordinate variables. They will hold the coordinate
  # information 
  times = ncfile.createVariable('time',np.float32,dimensions=['time',])
  lats = ncfile.createVariable('latitude',np.float32,dimensions=['latitude',])
  lons = ncfile.createVariable('longitude',np.float32,dimensions=['longitude',]) 
   
  try:
    times.units = time_out.units
  except:
    times.units = time_string
    
  try:
      times.calendar=time_out.calendar
  except:
      times.calendar=cal
  try:       
      lats.units = lat_out.units
  except:
      lats.units=lat_string
  try:
      lons.units=lon_out.units
  except:
      lons.units=lon_string
      
  # write data to coordinate vars.
  try:
      times[:] = time_out[:]
      
  except:
      times=time_out
      
  lats[:] = lat_out[:]
  lons[:] = lon_out[:]
  
  # now create variable(s)
  for d in varis.iteritems():
      var = ncfile.createVariable(d[0],np.float32,dimensions=\
                                  ['time','latitude','longitude'],\
                                  fill_value = mv)
      # write data to this variable
      try:
          var[:,:,:] = d[1][:,:,:]
          var.units = d[1].units
      except:# if not netcdf, need to read explicitly

          var[:,:,:] = d[1][0]
          var.units = d[1][1]   
     
  # now close file 
  ncfile.close() 
  
# Function to write netcdf object(s) --> netcdf objects 
def write_nc_1d(ofile,varis,lat_out,lon_out,lat_string="degrees_north",\
lon_string="degrees_east"):

  # note that lat_out, lon_out, time_out etc must 
  # contain full variable info (i.e. not just values)
  # varis is a dictionary of variable names: variables   
  ncfile=Dataset(ofile,"w")
  nrows = len(lat_out); ncols = len(lon_out)
  
  # create lat and lon dimensions
  ncfile.createDimension('latitude',nrows)
  ncfile.createDimension('longitude',ncols)
  # Define the coordinate variables. They will hold the coordinate
  # information 
  lats = ncfile.createVariable('latitude',np.float32,dimensions=['latitude',])
  lons = ncfile.createVariable('longitude',np.float32,dimensions=['longitude',]) 
   
    
  try:       
      lats.units = lat_out.units
  except:
      lats.units=lat_string
  try:
      lons.units=lon_out.units
  except:
      lons.units=lon_string
      
  # write data to coordinate vars.     
  lats[:] = lat_out[:]
  lons[:] = lon_out[:]
  
  # now create variable(s)
  for d in varis.iteritems():
      var = ncfile.createVariable(d[0],np.float32,dimensions=['latitude','longitude'],\
      fill_value = -999.99)
      # write data to this variable
      try:
          var[:,:] = d[1][:,:]
          var.units = d[1].units
      except:# if not netcdf, need to read explicitly
          var[:,:] = d[1][0]
          var.units = d[1][1]   
     
  # now close file 
  ncfile.close()
  
# Function to write netcdf with an arbitary first dimension (not time)
def write_nc_arb(ofile,varis,lat_out,lon_out,first_dim_name,first_dim_vals,\
first_dim_units,lat_string="degrees_north",lon_string="degrees_east",\
mv=-999.999):

  # Read in etc.
  ncfile=Dataset(ofile,"w")
  nrows = len(lat_out); ncols = len(lon_out); nfirst=len(first_dim_vals)
  
  # create lat and lon dimensions
  ncfile.createDimension(first_dim_name,nfirst)
  ncfile.createDimension('latitude',nrows)
  ncfile.createDimension('longitude',ncols)
   
  # Define the coordinate variables. They will hold the coordinate
  # information 
  first = ncfile.createVariable(first_dim_name,np.float32,dimensions=\
                                [first_dim_name,])
  lats = ncfile.createVariable('latitude',np.float32,dimensions=['latitude',])
  lons = ncfile.createVariable('longitude',np.float32,dimensions=['longitude',]) 
   
  # Do the lon/lats  
  try:       
      lats.units = lat_out.units
  except:
      lats.units=lat_string
  try:
      lons.units=lon_out.units
  except:
      lons.units=lon_string
      
  # Now the arbitary first dimension
  first.units=first_dim_units   
      
  # write data to coordinate vars.     
  lats[:] = lat_out[:]
  lons[:] = lon_out[:]
  first[:] = first_dim_vals
  
  # now create variable(s)
  for d in varis.iteritems():
      var = ncfile.createVariable(d[0],np.float32,dimensions=[first_dim_name,\
                                  'latitude','longitude'],\
      fill_value = mv)
      # write data to this variable
      try:
          var[:,:,:] = d[1][:,:,:]
          var.units = d[1].units
      except:# if not netcdf, need to read explicitly
          var[:,:,:] = d[1][0]
          var.units = d[1][1]   
     
  # now close file 
  ncfile.close()
  

def degrees_km_apart_lon(start,stop,lat,spacing_km):
 
    """
    Script takes a starting longitude at lat = lat and then creates a vector
    of points spacing_km apart along that latitude circle (until reaching stop)
    """
    start=np.radians(start); stop=np.radians(stop)
    # Convert spaing (km) to radians (NB. 2*pi radians in 2*pi*R km)
    circum=np.cos(np.radians(lat))*np.pi*2.*R
    dlon=2.*np.pi/circum*spacing_km # radians/spacing
    return np.degrees(np.arange(start,stop+dlon,dlon))

def degrees_km_apart_lat(start,stop,spacing_km):
 
    """
    Script takes a starting latitude at and then creates a vector of points spacing
    _km apart 
    """
    start=np.radians(start); stop=np.radians(stop)
    # Convert spaing (km) to radians (NB. 2*pi radians in 2*pi*R km)
    circum=np.pi*2.*R
    dlat=2.*np.pi/circum*spacing_km # radians/spacing
    return np.degrees(np.arange(start,stop+dlat,dlat))

 
# general time series functions
def movAvOdd(series,span):
    assert span % 2 != 0, "Span should be odd!"
    window = np.ones(span)/span
    output = np.zeros(series.shape)*np.nan
    # array formulation
    if len(series.shape)>1:
        ncols = series.shape[1]
        for col in range(ncols):
            smoothed = np.convolve(series[:,col],window,"valid")
            output[(span-1)/2:-(span-1)/2,col] = smoothed
    else: # vector 
        smoothed = np.convolve(series,window,"valid")
        output[(span-1)/2:-(span-1)/2] = smoothed
         
    return output
         
 
def movAvEven(series,seriesInd,span):
    assert span % 2 == 0, "Span should be even!"  
    window=np.ones(span)/span
    varOut=np.convolve(series,window,"valid")
    indOut=seriesInd[np.int(span/2.-1):-np.int((span/2.))]+0.5
    return varOut,indOut
 
def movVarEven(series,seriesInd,span):
    assert span % 2 == 0, "Span should be even!"
    varOut = np.zeros(len(series)-span+1) * np.nan
    for ii in range(span,len(series)+1):
        
        varOut[ii-span] = np.std(series[ii-span:ii])        
    indOut=seriesInd[span/2.-1:-(span/2.)]+0.5
     
    return varOut,indOut  
     
#def movVarianceEven(series,seriesInd,span):
#    assert span % 2 == 0, "Span should be even!"
#    varOut = np.zeros(len(series)-span+1) * np.nan
#    for ii in range(span,len(series)):
#        varOut[ii-span+1] = np.var(series[ii-span+1:ii+1])        
#    indOut=seriesInd[span/2.-1:-(span/2.)]+0.5
     
    return varOut,indOut  
     
def movKS(series,dist,span,ind):
     
    """
    Calculates the moving KS between series and normal/gamma distributions
    """
     
    assert dist == "norm" or dist == "gamma","'dist' must be either 'norm' or" + \
    "'gamma'"
     
    n = len(series)
    outData = np.zeros(n-span+1)
    outInd = np.zeros((outData.shape))
    evenPos = span/2.
    oddPos = (span+1)/2.
     
    if span % 2 == 0:
        even = True
      
    row = 0
    for ii in range(span,len(series)+1):
         
        if dist == "gamma":
            shape,loc,scale = stats.gamma.fit(series[ii-span:ii],floc=0)
            outData[row],p = stats.kstest(series[ii-span:ii],'gamma',\
            args=(shape,loc,scale))
             
        else:
            loc = np.mean(series[ii-span:ii])
            scale = np.std(series[ii-span:ii])
            outData[row],p = stats.kstest(series[ii-span:ii],'norm',\
            args=(loc,scale))    
 
        ind_sub = ind[ii-span:ii]         
         
        if even:
            outInd[row] = np.mean([ind_sub[evenPos],ind_sub[evenPos+1]]) 
        else:   
            outInd[row] = ind[oddPos]
        
        row+=1
 
    return outData,outInd
     
def DoY_stat(series,doy,span,stat):
    
    """
    This function takes a series and its corresponding DoY index (doy),
    and then calculates the statistic (stat - from the numpy - 'np' module) 
    of interest over a window of length span 
    
    Note: uses angles to compute distance - hence allowing 'wrap around' 
    near years' end
    """
    
    # Radians
    rad=np.radians(360/365.*doy)
    thresh=np.radians(span/365.*360)
    circ=2*np.pi
    
    # Associate function - making sure it's 'nan-compatible'
    stat="nan"+stat
    f = getattr(np,stat)
    
    # preallocate
    #out=np.zeros(len(series))*np.nan
    udoy=np.unique(doy)
    out=np.zeros(len(udoy))
    count=0
    for ii in udoy:
        logi=(np.abs(rad-ii)%circ)<=thresh
        out[count]=f(series[logi])
        count+=1
    
    return udoy,out
        
    
#def split_regression(x,y,thresh):
#    
#    """
#    This is a very simple function that finds the optimum 'knot' to 
#    fit a two-piece linear regression. 
#    
#    We simply iterate over all values of x and take the value that yields
#    thelowest RMSE. We then return the slopes of these different regression
#    lines. 
#    
#    We also return the regression model for the whole series (along with 
#    uncertainty in the regression coefficients for the whole model)
#    """
#
#    out=np.zeros((len(x)-2*thresh,4))*np.nan
#    mod=np.zeros(len(x))
#    count=0
#    for i in x:
#        below=x<=i
#        above=x>i
#        lx=x[below]; ly=y[below]
#        ux=x[above]; uy=y[above]
#        if len(lx)>thresh and len(ux)>thresh:
#            lp=np.polyfit(lx,ly,1)
#            up=np.polyfit(ux,uy,1)
#            mod[below]=np.polyval(lp,x[below])
#            mod[above]=np.polyval(up,x[above])
#            out[count,0]=i; out[count,1]=lp[0]; out[count,2]=up[0];
#            rmse=RMSE(mod,y);            
#            out[count,3]=rmse
#            count+=1
#            
#    maxr=np.nanmin(out[:,3])
#    x = sm.add_constant(x)
#    model = sm.OLS(y,x)
#    results = model.fit()
#    intercept,slope=results.params
#    errors=results.bse
#    idx=out[:,3]==maxr
#    row=out[idx,0]
#    x=x[:,1]
#    belowx=x[x<=row]
#    abovex=x[x>row]
#    abovey=y[x>row]
#    belowy=y[x<=row]
#    modbelow=np.polyval(np.polyfit(belowx,belowy,1),belowx)
#    modabove=np.polyval(np.polyfit(abovex,abovey,1),abovex)
#    xabove=sm.add_constant(abovex)
#    xbelow=sm.add_constant(belowx)
#    model_above = sm.OLS(abovey,xabove)
#    model_below = sm.OLS(belowy,xbelow)
#    results_above=model_above.fit()
#    results_below=model_below.fit()
#    
#    return out[idx,:],intercept,slope,errors,[belowx,modbelow],\
#    [abovex,modabove], results_below.bse[1], results_above.bse[1]
            
    
    

def critD(n,alpha):
     
    assert alpha == 0.01 or alpha == 0.05 or alpha == 0.10, "Not valid alpha!"
     
    if alpha == 0.01:
        k = 1.628
    elif alpha == 0.05:
        k = 1.358
    else:
        k = 1.224
     
    c = k/ (n**0.5 + 0.12 + 0.11/n**0.5)
     
    return c

def critical_r(x,y):
    
    """
    Takes two (possibly correlated) series (x,y) which may exhibit 
    autocorrelation. It then calculates significance whilst accounting for 
    this autocorrelation following Santer et al. (2000):

    N=n*(1-CD*CF)/(1+(CD*CF))

    where CD and CF are the respective lag 1 correlation coefficients.    
    
    This adjusted N is then used to calculate the variance of the residuals
    around the regression line relating x and y. Finally, adjusted N is used
    to look up the critical value for rejecting the null hypothesis. 
    
    """
    assert len(x) == len(y) and (np.sum(np.isnan(x)) + np.sum(np.isnan(y))) ==0
    
    # Fit trend line
    ps=np.polyfit(x,y,1)
    b=ps[0]
    #print "b:", b
    
    # Calculate residuals 
    e=np.polyval(ps,x)-y
    #print "e:",e
    
    # Calculate autocorrelation for residuals
    r=np.corrcoef(e[1:],e[:-1])[0][1]
    #print "r:",r
    
    # Adjusted n:
    neff=len(x)*(1.-r)/(1.+r)
    #print "Neff:", neff
    
    # Stdev of the residuals (with neff)
    se=np.sqrt(np.sum(np.power(e,2)) * 1./(neff-2))
    
    # Standard error of b
    denom=np.sqrt(np.sum(np.power((x-np.mean(x)),2)))
    sb=se/denom
    #print "sb: ", sb
    tb=b/sb
    #print tb
    # now evaluate probability
    p=stats.t.sf(tb,df=neff)  # exceedance probability
    if tb<0:
        p=1-p
        
    p=p*2 # two-tailed

    # return the correlation coefficient too
    rho=b*np.std(x)/np.std(y)
    return p,b,r,neff,rho 
     
def movStat(series,ind,obj,method,span):
    
    """
    This function calculates the moving statistic over an arbitary-length 
    window. It does this by simple iteration, so may be slow for large series.
    We access the particular method of the 'object' module by using 
    "getattr".
    
    """
    f = getattr(obj,method)
    n = len(series)
    outData = np.zeros(n-span+1)
    outInd = np.zeros((outData.shape))
    evenPos = span/2.
    oddPos = (span+1)/2.
    even=False
    row = 0
    if span % 2 == 0:
        even = True
    for ii in range(span,len(series)+1):
        outData[row] = f(series[ii-span:ii])
        ind_sub = ind[ii-span:ii]
        row +=1   
    if even:        
        ist=np.int(span/2.-1); istp=np.int(-(span/2.))
        outInd=ind[ist:istp]+0.5     
    else:
        ist=np.int((span-1)/2.); istp=np.int(-(span-1)/2.)
        outInd=ind[ist:istp]
    return outData,outInd


def doy_run_stat_grid(grid,doy,method,span,ndays,pctl=-9999,v=False,\
                      ret_anom=False):
    
    """
    Computes day of year statistic using a running window. 
    
    Input: 
        - grid: cube of data
        - doy: day of year 
        - method: statistic (from numpy module) to compute 
        - span: how many days in length should the window be? (MUST BE ODD)
        - ndays: how many days in the year? (365 or 366)
        - pctl: if given, it's the percentle (1-100)
        - v: verbose? If given, output after each loop iteration
        - ret_anom: return anomaly? If True, subtract the threshold from the 
        grid to yield anomaly
 
    Output: 
        - out: ndays x nrows x ncols of statistic output by "method"
        
    """
    assert span % 2 !=0, "Span should be odd!"   
    ntime,nrows,ncols=grid.shape
    out=np.zeros((ndays,nrows,ncols))
    f = getattr(np,method)
    # Convert days to radians
    rads=np.pi*2/np.float(ndays)*doy
    urad=np.unique(rads)
    thresh=(span-1)/2.* np.pi * 2/np.float(ndays)
    # Circle 
    c=np.pi*2 # Radians
    # Prealloacte if anomalies required
    if ret_anom:
        anom=np.zeros(grid.shape)
    
    # Loop over 1-->ndays and get centred stat
    for ii in range(len(np.unique(rads))):
        delta=np.abs(((rads-urad[ii])+np.pi)%c-np.pi)
        ind=delta<=thresh
        if pctl>=1: # Additional argument (percentile) expected...
            out[ii,:,:]=f(grid[ind,:,:],pctl,axis=0)
        else: # Assume no extra arguments...
            out[ii,:,:]=f(grid[ind,:,:],axis=0)

        if ret_anom:
            anom[ind,:,:]=grid[ind,:,:]-out[ii,:,:]
            
        if v:
            print "Finished iteration %.0f" % ii
            
    if ret_anom:
        return out,anom
    else:
        return out 
        
def doy_run_series(series,doy,method,span,ndays,pctl=-9999,v=False,\
                      ret_anom=False):
    
    """
    Computes day of year statistic using a running window. 
    
    Input: 
        - grid: cube of data
        - doy: day of year 
        - method: statistic (from numpy module) to compute 
        - span: how many days in length should the window be? (MUST BE ODD)
        - ndays: how many days in the year? (365 or 366)
        - pctl: if given, it's the percentle (1-100)
        - v: verbose? If given, output after each loop iteration
        - ret_anom: return anomaly? If True, subtract the threshold from the 
        grid to yield anomaly
 
    Output: 
        - out: ndays x nrows x ncols of statistic output by "method"
        
    """
    assert span % 2 !=0, "Span should be odd!"   

    out=np.zeros(ndays)
    f = getattr(np,method)
    # Convert days to radians
    rads=np.pi*2/np.float(ndays)*doy
    urad=np.unique(rads)
    thresh=(span-1)/2.* np.pi * 2/np.float(ndays)
    # Circle 
    c=np.pi*2 # Radians
    # Prealloacte if anomalies required
    if ret_anom:
        anom=np.zeros(len(series))
    
    # Loop over 1-->ndays and get centred stat
    for ii in range(len(np.unique(rads))):
        delta=np.abs(((rads-urad[ii])+np.pi)%c-np.pi)
        ind=delta<=thresh
        if pctl>=1: # Additional argument (percentile) expected...
            out[ii]=f(series[ind],pctl,axis=0)
        else: # Assume no extra arguments...
            out[ii]=f(series[ind],axis=0)

        if ret_anom:
            anom[ind]=series[ind]-out[ii]
            
        if v:
            print "Finished iteration %.0f" % ii
            
    if ret_anom:
        return out,anom
    else:
        return out     
    

     
def slidingCDF(series,seriesInd,span,step,percentiles,option):
    """
    NOTE: CDF is slightly different from percentiles (which are calculated 
    here; hence function should be renamed-'sliding percentiles' - but 
    name reatined here for legacy reasons)
     
    Vars require some explanation: 
    'series' is the sample which will have CDF evaluated using a sliding window
    'seriesInd' are the corrspondng time points (labels) at which 'series' was
    sampled. 
    'span' is the length of the sample
    'percentiles' is a vector of perceniles (in %)
     
    Outputted are the sliding percentiles
    """
    percentilesOut = np.zeros((len(series)-span+1,len(percentiles)))
    count1=0
    count2=0
    for ii in range(span-1,len(series),step):
         
        sample=series[ii-span+1:ii+1]
         
        count2=0
         
        if option=="gamma": # theoretical 
             
            shape_fit, loc_fit, scale_fit = stats.gamma.fit(sample)
 
            for jj in percentiles:
                percentilesOut[count1,count2]=\
                stats.gamma.isf(1-jj/100., shape_fit, loc_fit, scale_fit)
                 
                count2+=1
                 
             
        else: # empirical 
         
           for jj in percentiles:
               percentilesOut[count1,count2]=np.percentile(sample,jj)
               count2+=1
             
        # deal with counters     
        count1+=1
                 
    # deal with index
    if span % 2 == 0:
        add = 0.5
    else:
        add = 0
     
    indOut=seriesInd[(span/2.-1):-(span/2.)]+add
     
    return indOut,percentilesOut[:count1,:]
 
def probDist(data,dist,nsim,floc=True,slc=None):
    """
    This function fits 'dist' to 'data' and then evaluates the 
    probability that the data does indeed belong to 'dist'. This is 
    investigated by a Monte Carlo simulation with nsim realizations.
     
    NOTE: this function ONLY handles normal, gamma distributions, and 
    generalised extreme value distributions at present
     
    NOTE: The reference for this function is Clauset et al. (2009)
        - http://www.jstor.org/stable/pdf/25662336.pdf?acceptTC=true
    """
    assert dist == "gamma" or dist == "norm" or dist == "genextreme" or dist ==\
    "gumbel_r",\
    "'dist' must be either 'norm, 'gamma', 'genextreme', or 'gumbel_r' "
     
    # preallocate for MC runs
    dArray = np.zeros(nsim)
     
    # length of data array 
    ndata = len(data)
    # Get stat function
    statsdist=getattr(stats, dist)
    # start main
    if dist=="gamma" or dist =="genextreme": 
         
        if slc:
            shape=slc[0]; loc=slc[1]; scale=slc[2]
        elif floc:
            shape,loc,scale = statsdist.fit(data,floc=0) 
        else:
            shape,loc,scale = statsdist.fit(data) 
            
        d,p = stats.kstest(data,dist,args=(shape,loc,scale))
        distObj = statsdist(shape,loc=loc,scale=scale)
         
        # Monte Carlo
        for ii in range(nsim):
            # generate sample
            sample = distObj.rvs(ndata)
            # fit distribution
            if floc:
                shapeEst,locEst,scaleEst = statsdist.fit(sample,floc=0) 
            else:
                shapeEst,locEst,scaleEst = statsdist.fit(sample,shape=shape,\
                                                         loc=loc) 
            # evaluate fit with ks test
            dSamp,pSamp = stats.kstest(sample,dist,\
            args=(shapeEst,locEst,scaleEst))
            dArray[ii] = dSamp
 
    if dist == "gumbel_r":
        
        if slc:
            loc=slc[1]; scale=slc[-1]       
            loc,scale=statsdist.fit(data,loc=loc, scale=scale) 
    
        else:
            loc,scale = statsdist.fit(data) 
            
        d,p = stats.kstest(data,dist,args=(loc,scale))
        distObj = statsdist(loc=loc,scale=scale)    
        
        # Monte Carlo
        for ii in range(nsim):
            # generate sample
            sample = distObj.rvs(ndata)
            # fit distribution
            locEst,scaleEst = statsdist.fit(sample,loc=loc,scale=scale) 
            # evaluate fit with ks test
            dSamp,pSamp = stats.kstest(sample,dist,\
            args=(locEst,scaleEst))
            dArray[ii] = dSamp    
    
    else: # dist must be normal
        scale = np.std(data); loc = np.mean(data)
        # fit distribution          
        d,p = stats.kstest(data,'norm',args=(loc,scale))
        distObj = stats.norm(loc=loc,scale=scale)
         
        # Monte Carlo
        for ii in range(nsim):
            # generate sample
            sample = distObj.rvs(ndata)
            locEst = np.mean(sample); scaleEst = np.std(sample)
            # evaluate fit with ks test
            dSamp,pSamp = stats.kstest(sample,'norm',args=(locEst,scaleEst))
            dArray[ii] = dSamp
             
    # find fraction of darray is >=d
    dfrac = np.nansum(dArray>=d)/(nsim*1.0)*100.0  
 
    # plot if desired ('dist' fit to data and hist of data)                     
    return d,dArray,dfrac,distObj # returned value is in % for dfrac
     
         
def discreteKLD(full,part,dist):
    """
    Calculates the Kullback-Leibler divergence (see doi: 10.1016/j.procs.2012.04.098 )
    """
    assert dist=="gamma", "Function only set up for Gamma..."
   
    # fit gammas for both
    ashape,aloc,ascale = stats.gamma.fit(full,floc=0)
    bshape,bloc,bscale = stats.gamma.fit(part,floc=0)
    x=np.arange(np.min([np.min(full),np.min(part)]),
    np.max([np.max(full),np.max(part)])+1.)
    p_x=x**(ashape-1)*np.exp(-x/ascale)/(ascale**ashape*scipy.special.gamma(ashape))
    p_q=x**(bshape-1)*np.exp(-x/bscale)/(bscale**bshape*scipy.special.gamma(bshape)) 
    f = p_x*(p_x/p_q)
    KLD = scipy.integrate.simps(f,x)
     
    return KLD
     
#    # points at which pdf will be evaluated
#    x = np.linspace(0,np.max([np.max(a),np.max(b)]),1000)
#    
#    # get pdf
#    pa=stats.gamma.pdf(x,ashape,0,ascale)
#    pb=stats.gamma.pdf(x,bshape,0,bscale)
#    
#    pa = pa/np.sum(pa)
#    pb = pb/np.sum(pb)
#    # sum and output
#    return np.nansum(pa*(pa/pb))
     
def worstSample(series,seriesInd,span):
      
    out = np.zeros(len(series)-span+1)
    count1=0
 
    for ii in range(span-1,len(series)):
        sample=series[ii-span+1:ii+1]
        out[count1] = discreteKLD(series,sample,"gamma")
        count1+=1
         
   # deal with index
    if span % 2 == 0:
        add = 0.5
    else:
        add = 0
     
    indOut=seriesInd[(span/2.-1):-(span/2.)]+add    
 
    return out,indOut
     
     
def EDF (series,series_i):
    """
    Series and series_i should both be vectors.
    Function takes series and evaluates the cumulative distribution function.
    Non-exceedance probabilities are then interpolated at the series_i
    locations
    """
    # sort
    x=series[np.argsort(series)]
    ref=np.arange(len(series))
    # NE probs
    y=ref/(float(len(x)-1))*100.
    # interpolate
    yi = np.interp(series_i,x,y)
     
    return y,yi
     
def pred_intervals(x,y,xi):

    """ Computes the 95% confidence intervals around a linear regression 
    of y upon x"""
    
    n=np.float(len(x))
    yi_pred=np.polyval(np.polyfit(x,y,1),xi)
    yi=np.polyval(np.polyfit(x,y,1),x)
    se=1/(n-2)*np.sum(np.power(y-yi,2))
    xbar=np.mean(x)
    sy=1.96*np.sqrt(se*(1+1/n+np.power(xi-xbar,2)/np.sum(np.power(x-xbar,2))))
    
    return yi_pred-sy,yi_pred,yi_pred+sy

def GammaZ(series,val):
    """ 
    This function takes a series and fits a gamma distribution (with location
    fixed at zero). It then evaluates the cdf at val and converts this to a 
    z-score under the standard gaussian distribution. This z-score is returned,
    along with the cdf and parameters for the gamma distribution
    """  
    # fit gamma
    shape,loc,scale = stats.gamma.fit(series,floc=0)
         
    # evaluate cdf
    cdf = stats.gamma.cdf(val,shape,loc,scale)
     
    # get percent-point function for normal distribution given this val for
    # cdf
    z = stats.norm.ppf(cdf)
     
    return z,cdf, shape, scale
     
     
def ZGamma(series,z):
    """ 
    This function takes a series and fits a gamma distribution (loc=0). It 
    then evaluates the cdf for the standard normal distribution
    at z. This is used as input to the inverse cdf (i.e. quantile/ppf) using
    the fitted gamma distribution. Returned is the value which yields a 
    cdf for the fitted gamma equal to z under the standard normal.
    """
    # fit gamma
    shape,loc,scale = stats.gamma.fit(series,floc=0)
     
    # get cdf  of z under normal
    cdf = stats.norm.cdf(z)
     
    # get ppf/quantile of cdf under gamma
    quantile = stats.gamma.ppf(cdf,shape,loc=0,scale=scale)
     
    return quantile
    
    
def biasCorr_s(ref_series,ref_years,mod_series,mod_years,proj_st,proj_stp):
    
    """
    This function applies a simple bias corrections routine that assumes 
    distributions of the same shape. The correction is:
    
    X' = mu_obs(cal) + SD_obs(cal)/SD_mod(cal)*(mod(proj)-mu_sim(obs))
    
    The ref_series and corresponding ref_years are used to define the 
    calibration period and years; data from the mod_series are selected
    from these years and the correction calculated. The correction is then 
    applied to the whole mod_series.
    
    Input: 
        - ref_series: observations upon which to base correction
        - ref_years: the time values of the ref_series
        - mod_series: the modelled series we wish to bias correct
        - mod_years: the time values of the mod_series
    Output: 
        - corrected: the bias corrected modelled series for times given by 
          >=proj_st & <=prj_stp
    """
    # Check for common years...
    assert (np.sum(mod_years==ref_years[0]) > 0 and \
    np.sum(mod_years==ref_years[-1])>0), \
    "Mod series doesn't run for full calibration period!"
    
    # Identify the common block for calibration 
    mod_cal = mod_series[np.logical_and(mod_years>=ref_years[0],\
    mod_years<=ref_years[-1])]
 
    # Identify the part of the modelled series that we want to provide 
    # projections for
    mod_proj = mod_series[np.logical_and(mod_years>=proj_st,mod_years<=proj_stp)]   
        
    corrected = np.mean(ref_series) + np.std(ref_series)/np.std(mod_cal) * \
    (mod_proj-np.mean(mod_cal))
    
    return corrected
    
def changeFactor(ref_series,ref_years,mod_series,mod_years,proj_st,proj_stp):    
         
    """
    This function applies a simple change factor bias correction:

    X' = mu_mod(proj) + std_mod(proj)/std_mod(obs) * (ref_obs-mu_mod(ref))
    
    The ref_series and corresponding ref_years are used to define the 
    calibration period and years; data from the mod_series are selected
    from these years and the correction calculated. The correction is then 
    applied to the part of the mod series >=proj_st & <=prj_stp.
    
    Input: 
        - ref_series: observations upon which to base correction
        - ref_years: the time values of the ref_series
        - mod_series: the modelled series we wish to bias correct
        - mod_years: the time values of the mod_series
    Output: 
        - corrected: the bias corrected modelled series for times given by 
          >=proj_st & <=prj_stp
    """
    assert (np.sum(mod_years==ref_years[0]) > 0 and \
    np.sum(mod_years==ref_years[-1])>0), "Mod series doesn't run for full calibration period!"   
    
    # Check for common years...
    assert (np.sum(mod_years==ref_years[0]) > 0 and \
    np.sum(mod_years==ref_years[-1])>0), \
    "Mod series doesn't run for full calibration period!"
    
    # Identify the common block for calibration 
    mod_cal = mod_series[np.logical_and(mod_years>=ref_years[0],\
    mod_years<=ref_years[-1])]
 
    # Identify the part of the modelled series that we want to provide 
    # projections for
    mod_proj = mod_series[np.logical_and(mod_years>=proj_st,mod_years<=proj_stp)]  
    
    # Apply the correction
    corrected = np.mean(mod_proj) + np.std(mod_proj)/np.std(mod_cal) * \
    (ref_series-np.mean(mod_cal))
    
    return corrected
    
def bias_Corr_s_mu(obs_cal,mod_cal,proj):

    """
    This function applies a simple change factor bias correction:
    
    This function differs from bias_Corr_s in that it only deals with mean 
    changes
    
    Input: 
        - obs_cal: observations upon which to base correction
        - mod_cal: the modelled series we wish to bias correct
    Output: 
        - corrected: the bias corrected modelled series (mean) for proj

    Notes:
        - This equation is applied:
        mu_proj = mu_obs(cal) + SD_obs(cal)/SD_mod(cal)*(mu_mod(proj)-mu_mod(obs))
    """

    mu_proj = np.nanmean(obs_cal,axis=0)+np.nanstd(obs_cal,axis=0)/np.nanstd(mod_cal,axis=0)*\
    (np.nanmean(proj,axis=0)-np.nanmean(mod_cal,axis=0))      
    
    return mu_proj
    
def changeFactor_mu(obs_cal,mod_cal,proj):    
         
    """
    This function applies a simple change factor bias correction:
    The function differs from changeFactor in that it only deals with mean 
    changes:

    Input: 
        - obs_cal: observations upon which to base correction
        - mod_cal: the modelled series we wish to bias correct
    Output: 
        - corrected: the change-factor-corrected modelled series (mean) for proj

    Notes:
        - This equation is applied:
        
    mu_proj = mu_mod(proj)+std_mod(proj)/std_mod(obs)*(mu_obs(obs)-mu_mod(obs))
    """
    mu_proj=np.mean(proj,axis=0)+np.std(proj,axis=0)/np.std(mod_cal,axis=0)*\
    (np.mean(obs_cal,axis=0)-np.mean(mod_cal,axis=0))
    
    return mu_proj
    
         
    
# Empirical Quantile-Mapping (matching) function
def QQmatch(obs,sim,sim_i,detrend_i=True):
     
    # Check same number of rows
    assert len(obs)==len(sim), "'obs' and 'sim' must be same length!"
     
    # detrend proj if desired
    if detrend_i:
        # get trend
        ps = np.polyfit(np.arange(len(sim_i)),sim_i,1)
        # subtract trend 
        sim_i = np.mean(sim_i) + signal.detrend(sim_i)
     
    # indices of sorted array    
    indObs = np.argsort(obs)
    indSim = np.argsort(sim)
    position = np.arange(len(obs))
 
    # define correction factors; cfs must be added to each quantile
    cfs = obs[indObs]-sim[indSim]
    simCorr = (sim[indSim] + cfs)
     
    # now bias correct sim_i
    # sim_i < min(sim) | sim_i > max(sim) are corrected by min(cfs) and 
    # max(cfs), respectively.
    simCorr_i = sim_i
    simCorr_i[simCorr_i > np.max(sim)] = simCorr_i[simCorr_i>np.max(sim)] + cfs[-1]
    simCorr_i[simCorr_i < np.min(sim)] = simCorr_i[simCorr_i<np.min(sim)] + cfs[0]    
     
    # index of remaining
    logi = np.logical_and(simCorr_i>=np.min(sim),simCorr_i<=np.max(sim))
    
    # interpolate those within bounds (note that simCorr_i is time-ordered
    # after call)
    simCorr_i[logi] = np.interp(simCorr_i[logi],sim[indSim],simCorr)
     
    # finally, add trend back in, if necessary: 
    if detrend_i:
        simCorr_i = (simCorr_i - np.mean(simCorr_i)) + \
        ps[0]*np.arange(len(simCorr_i)) + np.interp(ps[1],sim[indSim],simCorr)
         
    return simCorr[np.argsort(position[indSim])],simCorr_i
 
# Variant of empirical Quantile-Mapping (matching) function - linearly 
# extrappolates correction factor
def QQmatch2(obs,sim,sim_i):
     
    """obs = self explanatory
        sim = hindcast
        sim_i = projection points
    """
        
    # Check same number of rows
    assert len(obs)==len(sim), "'obs' and 'sim' must be same length!"
         
    # indices of sorted array    
    indObs = np.argsort(obs)
    indSim = np.argsort(sim)
    position = np.arange(len(obs))
 
    # define correction factors; cfs must be added to each quantile
    cfs = obs[indObs]-sim[indSim]
    simCorr = (sim[indSim] + cfs)
    # simCorr_i = sim_i*1.
    # define linear function: cfs as a function of sim
    ps = np.polyfit(sim[indSim],cfs,1)
    # get goodness of fit
    r=(stats.pearsonr(sim[indSim],cfs)[0])**2
     
    # index of those within bounds -
    logi = np.logical_and(sim_i>=np.min(sim),sim_i<=np.max(sim))
    # index of those out of bounds -
    logi_x = np.logical_or(sim_i<np.min(sim),sim_i>np.max(sim))
 
    # interpolate those within bounds (note that sim_i is time-ordered
    # after call)
    sim_i[logi] = np.interp(sim_i[logi],sim[indSim],simCorr)
    # for those out of bounds, add cf which is obtained via polynomial 
    sim_i[logi_x] = sim_i[logi_x]+np.polyval(ps,sim_i[logi_x])
    
    # return    
    return simCorr[np.argsort(position[indSim])],sim_i,r    
    
def QQmatch3(obs,sim,pctls,sim_i,extrap=False):
     
    """
        Differs from QQmatch3 in that it evaluates correction at a series of 
        pre-determined percentiles - not determined by the data. This is required
        if obs and sim are different lengths. Note that this function is 'best' 
        out of QQ1/2 as both extrapolation + constant treatment is possible 
        (through extrap keyword)
        
        obs = self explanatory
        sim = hindcast
        sim_i = projection points
        pctls = numpy array of percentiles (increasing order)
        extrap = if true, linear extrapolation is performed outside obs range
        
    """

    # get percentiles of obs/sim
    obs_x=np.nanpercentile(sim,pctls) # transform from 
    obs_y=np.nanpercentile(obs,pctls) # transform to 
    simCorr=np.zeros(sim_i.shape)*np.nan

    # interpolate sim_i 
    in_bounds=np.logical_and(sim_i>=np.min(obs_x),sim_i<=np.max(obs_x))
    simCorr[in_bounds]=np.interp(sim_i[in_bounds],obs_x,obs_y) # in bounds
    
    if extrap: # linear functions
        out_bounds=np.logical_or(sim_i<np.min(obs_x),sim_i>np.max(obs_x)) 
        ps=np.polyfit(obs_x,obs_y,1) # difference in percentiles
        simCorr[out_bounds]=np.polyval(ps,sim_i[out_bounds])
        
    else: # constant correction
        above=sim_i>np.max(obs_x)
        below=sim_i<np.min(obs_x)
        deltas=obs_y-obs_x
        simCorr[above]=sim_i[above]+deltas[-1] # add diff between top percentile
        simCorr[below]=sim_i[below]+deltas[0] # add diff between bottom percentile
                    
    # return    
    return simCorr
     
def dailyCounter(yrStart,monStart,dayStart,yrEnd,monEnd,dayEnd):
    start=datetime.datetime(yrStart,monStart,dayStart)
    end=datetime.datetime(yrEnd,monEnd,dayEnd)
    n = (end-start).days+1
    yr = np.zeros(n); mon = np.zeros(n); day=np.zeros(n)
    present=start
    count=0
    dates=[]
    while present <= end:    
        yr[count] = present.year; mon[count] = present.month
        day[count] = present.day       
        present = present + datetime.timedelta(days=1) 
        count += 1
        dates.append(present)
      
    return dates,yr,mon,day
     
def annualSums(yrVect,series):
    uYrs = np.unique(yrVect)
    annual = np.zeros((len(uYrs),2))*np.nan
    annual[:,0]=uYrs
    count=0
    for ii in uYrs:
        annual[count,1]=np.nansum(series[yrVect==ii])
        count+=1
    return annual
     
def RMSE(obs,sim):
    rmse=(np.sum((obs-sim)**2)/len(sim))**0.5
    return rmse
     
def slidingCorrel(seriesArray,target,span,rtype):
     
    """ 
    Takes array and calculates correlation with 'target' using 
    a sliding window of length 'span'. All possible correlations
    are computed. i.e starting at the span^th data point and going to
    the last point. Hence, in an n by m array, there will be 
    n-span+1 by m correlation coefficients
     
    NB.
        rtype     :     "pearson" or "spearman"; determines the 
                         correlation function applied
    """
    assert rtype=="pearson" or rtype == "spearman","rtype must be 'pearson' or 'spearman'!"
     
    nrows=seriesArray.shape[0]
    ncols=seriesArray.shape[1]
    outCorrel=np.zeros(seriesArray.shape)*np.nan
     
    for col in range(ncols):
         
        for row in range(span,nrows+1):
             
            sample = seriesArray[row-span:row,col]
            sampleTarget = target[row-span:row]
             
            # correlate
            if rtype == "spearman":
                 
                # Spearman will calculate r if NaN is present, so stop this here
                if np.sum(np.isnan(sample))==0 and \
                np.sum(np.isnan(sampleTarget))==0:
                    outCorrel[row-1,col]=(stats.spearmanr(sample,sampleTarget))[0]
                 
            else:
                 
                outCorrel[row-1,col]=(stats.pearsonr(sample,sampleTarget))[0]
    print row
                 
    return outCorrel             
     
def plotRegion(lons,lats,proj,ax):

    lons[lons>180.]=lons[lons>180.]-360.
    llcrnrlat=np.min(lats) - (np.max(lats)-np.min(lats))*0.3
    urcrnrlat=np.max(lats) + (np.max(lats)-np.min(lats))*0.3
    llcrnrlon=np.min(lons) - (np.max(lons)-np.min(lons))*0.3
    urcrnrlon=np.max(lons) + (np.max(lons)-np.min(lons))*0.3
     
    # go anticlockwise from bottom left (min lon,min lat) to draw box
    coordslat=[np.min(lats),np.min(lats),np.max(lats),np.max(lats),np.min(lats)]
    coordslon=[np.min(lons),np.max(lons),np.max(lons),np.min(lons),np.min(lons)]
    m =\
Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,\
urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,projection=proj,\
resolution='i',ax=ax)
    m.drawcoastlines(linewidth=0.5)
    m.bluemarble()
 
    # convert to map projection 
    lonplot,latplot=m(coordslon,coordslat)
     
    # plot   
    mapHandle=m.plot(lonplot,latplot,color="red")

     
    return m
 
 
def spherical2CartGrad(lonGrid,latGrid,dfdlon,dfdlat):
    """
    Takes arrays of derivatives calculated with respect to lon/lat
    and converts to dx/dy
    NOTE: input should be in degrees!
    Output is df/dKilometre
    """
    coefLat=1. * 360./(2 * np.pi * R)
    coefLon=coefLat*np.cos(np.radians(latGrid))
     
    dfdx=dfdlon*coefLon; dfdy=dfdlat*coefLat   
 
    return dfdx,dfdy
 
def seasonMean(year,month,series,season,start=-np.inf,stop=np.inf):
     
    """
    *****
    Outputs seasonal means. 
    *****
    Input
        - year = vector of years(len=len(series))
        - month = vector of months(len=len(series))
        - series = the MONTHLY time series to be averaged
        - season = conventional 3-letter abbreviation for season (LOWER CASE)
        - start = earliest year in desired output
        - stop = latest year in desired output
    Output
        - means = seasonal means
        - yrOut = the years corresponding to the seasonal means
    """
     
    possSeasons = ["djf","mam","jja","son"]
    corrMons = [(12,1,2),(3,4,5),(6,7,8),(9,10,11)]
     
    # check
    assert season in possSeasons,"season should be conventional 3-month code!"
    mons = [corrMons[ii] for ii in range(4) if season == possSeasons[ii]][0]
     
    # index of last month in season
    endMons = np.where(month==mons[-1])[0]
     
    # check first season is fully represented
    if endMons[0]==1:
        endMons = endMons[1:]        
    elif month[endMons[0]-2] != mons[0]: # first season is not complete
        endMons = endMons[1:]
         
    # now cleverly get means
    means = (series[endMons] + series[endMons-1] + series[endMons-2])/3.
    yrOut = year[endMons]
    return means[np.logical_and(yrOut>=start,yrOut<=stop)], \
    yrOut[np.logical_and(yrOut>=start,yrOut<=stop)]
        
     
def deltaPmuV(years,series,critValue,span,distribution):
    """
    This function permits exploration as to whether changes in variance or
    mean drive the changing probability of an event. Only gamma and normal 
    distributions are allowed at this point in time
    """
    assert distribution == "norm" or distribution == "gamma", "Distribution" +\
    " must be 'norm' or 'gamma' only"
     
    mu = np.nanmean(series)
#    print mu
    std = np.nanstd(series)
#    print std
    varRescaled = np.zeros((len(series)-span+1)) * np.nan
    muRescaled = np.zeros((len(series)-span+1)) * np.nan
    orig = np.zeros((len(series)-span+1)) * np.nan
     
    varRescaled_z = np.zeros((len(series)-span+1)) * np.nan
    muRescaled_z = np.zeros((len(series)-span+1)) * np.nan
    orig_z = np.zeros((len(series)-span+1)) * np.nan
     
    for ii in range(span,len(series)):
         
        samp = series[ii-span:ii] 
         
        # rescale var
        sampVarRes = np.mean(samp) + (samp-np.mean(samp))*std/np.std(samp)
         
        # rescale mean
        sampMuRes = samp + (mu-np.mean(samp))
         
        # now evaluate p of exceedance for this series 
        # (this is where we need distributions)
         
        if distribution == "gamma":
            # z,cdf, shape, scale
            zVarRes,cdfVarRes,shapeVarRes,scaleVarRes = GammaZ(sampVarRes,critValue)
            pVarRes = (1-cdfVarRes)*100
             
            zMuRes,cdfMuRes,shapeMuRes,scaleMuRes = GammaZ(sampMuRes,critValue)
            pMuRes = (1-cdfMuRes)*100
             
            zOrig,cdfOrig,shapeOrig,scaleOrig = GammaZ(samp,critValue)
            pOrig = (1-cdfOrig)*100
             
        else:
             
            pVarRes = (1-stats.norm.cdf(critValue,loc=np.mean(sampVarRes),\
            scale = np.std(sampVarRes)))*100
            zVarRes = (critValue-np.mean(sampVarRes))/np.std(sampVarRes)
             
            pMuRes = (1-stats.norm.cdf(critValue,loc=np.mean(sampMuRes),\
            scale = np.std(sampMuRes)))*100   
            zMuRes = (critValue-np.mean(sampMuRes))/np.std(sampMuRes)
             
            pOrig = (1-stats.norm.cdf(critValue,loc=np.mean(samp),\
            scale = np.std(samp)))*100  
            zOrig = (critValue-np.mean(samp))/np.std(samp)
         
         
        varRescaled[ii-span] = pVarRes
        muRescaled[ii-span] = pMuRes
        orig[ii-span] = pOrig
         
        varRescaled_z[ii-span] = zVarRes
        muRescaled_z[ii-span] = zMuRes
        orig_z[ii-span] = zOrig
         
    return years[~np.isnan(varRescaled)],varRescaled[~np.isnan(varRescaled)], \
    muRescaled[~np.isnan(varRescaled)], orig[~np.isnan(varRescaled)], \
    varRescaled_z[~np.isnan(varRescaled)], muRescaled_z[~np.isnan(varRescaled)],\
    orig_z[~np.isnan(varRescaled)]
 
def r2T(r,df):
    """
    This function takes a Pearson's r-value and converts to the Student's 
    t-statistic. It also returns the two-tailed probability that t = 0.
     
    Input:
        - r = Pearson's correlation coefficient
        - df = degrees of freedom (n-2)
    Output:
        - t = Student's t-statistic
        - p = two-tailed probability that t = 0
    NB. Formula r --> t taken from:
        - https://en.wikipedia.org/wiki/Pearson_product-\
        moment_correlation_coefficient
        --- valid for bivariate normal distribution, but relatively robust for 
            non-normal distribution
    """
     
    t = r * np.sqrt(df/(1-np.square(r)))
    p = 2*(1-np.abs(stats.t.cdf(t,df)))
    return t,p

def conTimes(time_str="",calendar="",times=np.empty(1),safe=False):

    """
    This function takes a time string (e.g. hours since 0001-01-01 00:00:00),
    along with a numpy array of times (hours since base) and returns: year, month, day.
	
    Inefficient at present (with mulitple vectorize calls), but probably sufficient
    """

    timeObj=utime(time_str,calendar=calendar)

    # set up the functions
    f_times = np.vectorize(lambda x: timeObj.num2date(x))
    f_year = np.vectorize(lambda x: x.year)
    f_mon = np.vectorize(lambda x: x.month)
    f_day = np.vectorize(lambda x: x.day)
    f_hour = np.vectorize(lambda x: x.hour)
    
    # Call them
    pyTimes = f_times(times); year = f_year(pyTimes); mon = f_mon(pyTimes)
    day = f_day(pyTimes); hour = f_hour(pyTimes)
    
    # check that 'proper' datetime has been returned:
    if safe:
        year=np.atleast_1d(year); mon=np.atleast_1d(mon); day=np.atleast_1d(day)
        pyTimes = np.array([datetime(year[ii],mon[ii],day[ii]) for ii in \
        range(len(year))])

    return year,mon,day,hour,pyTimes		

 
#def sigSmoothed(y):
#     
#    """
#    This function takes a time series (y), and calculates the trend (b)
#    The standard error of the trend (from standard theory) is:
#     
#                        sb = se/(sum(t-t')^2)^0.5
#     
#    t is the time index (overbar=mean); se is the residuals of the variance a
#    round the regression line, given by:
#     
#                      se^2 = 1/ (n-2) * sum(e(t)^2)
#                 
#    in which e(t)^2 is (y-mean(y))          
#     
#    the test statistic is then obtained:
#                 
#                               t = b/sb
#                                
#    It follows a t-distribution with n-2 degrees of freedom. The above is
#    outlined in detail here: 
#    http://www.arl.noaa.gov/documents/JournalPDFs/SanterEtal.JGR2000.pdf
#     
#    the degrees of freedom also need to be updated to reflect reduced n
#     
#    """
#     
#    # n
#    n = len(y)
# 
#    # x 
#    x = np.arange(n)
# 
#    # fit linear regression
#    ps = np.polyfit(x,y,1)
#    # get residuals
#    residuals = np.polyval(ps,x)-y
#    # get lag 1 acf from residuals 
#    r = sm.tsa.stattools.acf(residuals,nlags=1)[1]   
#    # effective degrees of freedom; r is the lag 1 acf  
#    n_eff = n * (1-r)/(1+r)
#    # residual variance
#    se = np.sqrt(1/(n_eff-2) * np.sum(np.square(residuals)))
#    # standard error
#    sb = se/np.sqrt(np.sum(np.square((x-np.mean(x)))))
#    # ratio (follows t-distribution)
#    tb = ps[0]/sb
#    # adjust df using n_eff
#    df = n_eff-2
#    # finally, get two-tailed significance of tb
#    p = stats.t.sf(np.abs(tb),df) * 2
#     
#    return ps[0],p, tb
    
def mooresOff(delta):
    """ 
    This function finds the list of offsets required to find the Moores set - 
    that is the 'ring' of values surrounding a cell already identified as 
    being on the boundary. Input (delta) is the step just taken to land in the 
    Moores set (e.g. -1,0 to go from p to p-1,0). First element of delta is 
    delta row, second is delta column
    """
    
    # determine the list of clockwise neighbours based on the direction of the
    # backtrace
    
    top=np.array([[0,1],[1,0],[1,0],[0,-1],[0,-1],[-1,0],[-1,0],[0,1]])
    right=np.array([[1,0],[0,-1],[0,-1],[-1,0],[-1,0],[0,1],[0,1],[1,0]])
    bot=np.array([[0,-1],[-1,0],[-1,0],[0,1],[0,1],[1,0],[1,0],[0,-1]])
    left=np.array([[-1,0],[0,1],[0,1],[1,0],[1,0],[0,-1],[0,-1],[-1,0]])
    
    if delta[0]==-1 and delta[1]==0:
        m=top
    elif delta[0]==0 and delta[1]==1:
        m=right
    elif delta[0]==1 and delta[1]==0:
        m=bot
    elif delta[0]==0 and delta[1]==-1:
        m=left
    else:
        raise ValueError\
        ("Whoops, backtrace with unexpected identity! [dr,dc = %.0f,%.0f]" %\
        (delta[0],delta[1]))
    
    return m

def trace(mask,rows,cols):
    
    """
    This function scans the mask (bottom-->top; left-->right) until it meets
    a True element which is the boundary we're after. 
    
    It then implements the Moores Neighbour Trace Algorithm to trace 
    the border. 
    """
    nrows,ncols=mask.shape
    # preallocate for the output border    
    out=np.zeros((nrows*ncols*100,2))
        
    # scan from bottom to top, left to right, until we meet a "True" element
    p = np.zeros(2)
    rlst=nrows-1
    clst=0
    trigger=False
    for r in range(nrows-1,0,-1):
        for c in range(0,ncols):
            if mask[r,c]:
                p[0]=rows[r]; p[1]=cols[c]
                trigger=True
                break            
            rlst=r; clst=c
        if trigger == True:
            break

    # deltas which yield backtrace
    dr=(rlst-r); dc=(clst-c)
    
    # -- -- -- The starting pixel will have been identified -- -- -- #   
    out[0,0]=rows[r]; out[0,1]=cols[c]   
    # backtrace (after preallocation)
    b=np.zeros(2)
    b[0]=p[0]+dr; b[1]=p[1]+dc
    
    # determine the list of clockwise neighbours based on the direction of the
    # backtrace        
    m = mooresOff(np.array([dr,dc]))
    c=b+m[0,:]
    
    # Reminder:
    # m = Moores nerighbours (increments)
    # b = backtraced cell
    # c = clockwise neighbour (one of the Moores set)    
    # p = the cell at the center of the Moores set
    # s = first p
     
    new_dr=np.nan; new_dc = np.nan
    count = 1 # note starts at 1 because one entry already in 'out'
    step=1
    s=p*1.
    while True:
        
        if mask[c[0],c[1]]:    
            
            # border cell located
            out[count,0]=rows[c[0]]; out[count,1]=cols[c[1]] # store
            p = c # replace p with c
            c = c-m[step,:] # backtrack
            
            # reset the list of increments to clockwise neighbours (m) 
            m=mooresOff(m[step,:]*-1)  
                           
            # now reset the step
            step=0
        
            # increment count
            count +=1
            
        else:
            
            delta=c-(c+m[step,:])
            new_dr=delta[0]; new_dc=delta[1]
            c = c+m[step,:]
            step+=1
            
        # two options for breaking: 
        # 1) same entry direction and start direction  
        if (p==s).all()==True and count !=1:
            break
        
        #if (p==s).all()==True and new_dr==dr and new_dc == dc and count !=1 :
            
           # break
        # 2) we've run round the grid many times!
        #elif count > (nrows*ncols) and (p==s).all()==True:
            
           # break
 
    return out,count

def haversine(point1, point2, miles=False):
    
    """ 
    Calculate the great-circle distance bewteen two points on the Earth surface.

    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.

    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))

    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.

    """
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * R * np.arcsin(np.sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers
    
def haversine_fast(lat1,lng1,lat2,lng2,miles=False):
    
    """ 
    Calculate the great-circle distance bewteen two points on the Earth surface.

    :input: two scalars (lat1,lng1) and two vectors (lat2,lng2)

    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))

    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.

    """
    # convert all latitudes/longitudes from decimal degrees to radians
    lat1=np.radians(lat1); lat2=np.radians(lat2); lng1=np.radians(lng1); 
    lng2=np.radians(lng2)

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * R * np.arcsin(np.sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers


#def earthDist(lonlat,point):
#    """ Convenience function to calculate the Earth-surf distance between
#    a point and an array of lon/lats"""
#    # to radians
#    lonlat=np.radians(lonlat)
#    point=np.radians(point)
#    # haversine formula
#    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
#    h = 2 * R * np.arcsin(np.sqrt(d))
    

def optM(data,minM,maxM):
   
    """
    Function uses Bayesian principles to compute the optimum number of evenly
    spaced bins for a histogram.
    
    Input:
    
    -data: the time series to be considered
    -minM: the minimum number of bins to be tried
    -maxM: the maximum number of bins to be tried
   
    Output:
    
    -optM: the optimum number of evenly-spaced bins
    -bins: the edges of the optM bins
   
    Notes:
    
    -Algorithm taken from Matlab implementation reported here:
    http://huginn.com/knuth/papers/knuth-histo-draft-060221.pdf
   
    -In general, a brute-force approach is adopted, such that every number of
    bins from minM to maxM is tried. The value of M which maximises the log
    posterior likelihood is retained
   
    -No error checking is performed
    
    """
   
    ms=np.arange(minM,maxM)
    n=np.float(len(data))
    logp = np.zeros((len(ms),2))
    logp[:,0]=ms
    count=0
    for M in ms:
        nk = np.array(np.histogram(data,bins=M)[0])
        logp[count,1] = n*np.log(M) + gammaln(M/2.) - gammaln(n+M/2.) - \
       M*gammaln(0.5) + np.sum(gammaln(nk+0.5));
        count+=1
       
    optM=logp[logp[:,1]==np.max(logp[:,1]),0][0]
    pmf,bins=np.histogram(data,optM)
    return optM,bins
   
def pmf(data,bins):
   
    """
    Function uses a histogram to calculate the probability mass function
    for respective bins.
    
    Input:
   
    -data: the time series to be considered
    -bins: the edges of the bins (note that optM provides this in output)
   
    Output:
    
    -pmf: the pmf for each of the resepctive bins
    -err: the uncertainty on the respective pmfs
   
    Notes:
    
    -Formulas to calculate pmf & var are taken from
    http://huginn.com/knuth/papers/knuth-histo-draft-060221.pdf
   
    -No error checking is performed
    """
    n=len(data)
    M=len(bins)-1
    nk = np.histogram(data,bins)[0]
    pmf=(nk+0.5)/(n+M/2.)
    err=(nk+0.5)*(n-nk+(M-1)/2.)*(n+M/2.+1)**(-1)*(n+M/2.)**(-2)
   
    return pmf,err
    
def stripNC(folder,matchstr1,matchstr2,xcmd,step,oname,var,v=False):
    
    """
    Function "thins" a set of .nc files that might otherwise be too large
    to process in a single merged request. 
    
    Input:
        - folder: input directory holding the .nc files
        - matchstr: a string used for patern matching to select a subsample 
        of .nc files in the folder
        - step: how many timesteps should be skipped. For example, '2' means 
        take every second timestep; '10' every 10, etc...    
         
    Output:
        - oname: a merged netcdf file (should include path)
        
    Note:
        - writes all temporary files to /tmp. Cleans them up afterwards. 
         
    """
    files=[ii for ii in os.listdir(folder) if matchstr1 in ii and ".nc" in ii and \
    matchstr2 in ii]
    os.system("export MAX_ARGC=10000")
    
    assert len(files)>0, "'Files' is empty!"
    if v:
        for ii in files: print "Will process: %s" % ii   
    
    c=0
    for f in files:
        data=Dataset(folder+f,'r')
        nt,nr,nc=data.variables[var].shape
        ind=",".join(map(str,range(1,nt,step)))
        cmd="cdo -L -s --no_warnings seltimestep,%s %s %s /tmp/scratch_%.0f.nc" % \
        (ind,xcmd,folder+f,c)
        fail=os.system(cmd); assert fail==0,"\nThis CDO call failed:\n%s" % cmd
        c+=1
        if v:
            print "\t[*update: processed file: %s in GeneralFunctions module...]" %\
            f
    if v:
        print "loop complete"    #  Now merge
    cmd="cdo -L -s --no_warnings mergetime /tmp/scratch*.nc %s" %(oname)   
    fail=os.system(cmd); assert fail==0,"\nThis CDO call failed:\n%s" % cmd
    print "Saved %s" % oname
    trash=os.listdir("/tmp/")
    for ii in trash:
        if "scratch" in ii and ".nc" in ii:
            os.remove("/tmp/"+ii)
    os.system("trash-empty")
    return 0
        
def HeatIndexNWS_eqns(t,rh,crit,c):
    
  """ Contains the equations to be called from the main NWS algorithm, below"""
                                                           # NWS practice
  HI = (0.5*(t+61.0+((t-68.0)*1.2)+(rh*0.094)) + t)*0.5    # avg (Steadman and t)
  HI[t<=40] = t[t<=40]     
                            
  #testGrid=np.ones(HI.shape)# for debugging...

  if (np.all(t<crit[0])):
      pass
  else:
      loc1=HI>=crit[0]
      HI[loc1]=\
                c[0]+ c[1]*t[loc1] + c[2]*rh[loc1] + c[3]*t[loc1]*rh[loc1]  + \
                c[4]*t[loc1]**2+c[5]*rh[loc1]**2.+c[6]*t[loc1]**2.*rh[loc1]+\
                c[7]*t[loc1]*rh[loc1]**2+c[8]*t[loc1]**2.*rh[loc1]**2.
                 
      loc2=np.logical_and(np.logical_and(rh<13.,t>80.),t<112.)
      HI[loc2]=\
      HI[loc2]-((13.-rh[loc2])/4.)*np.sqrt((17.-np.abs(t[loc2]-95.))/17.)

      
      loc3=np.logical_and(np.logical_and(rh>85.,t>80.),t<87.)
      HI[loc3]=\
      HI[loc3]+((rh[loc3]-85.)/10.)*((87.-t[loc3])/5.)      

  return HI
  
def HeatIndexNWS(t,rh,opt=False,iounit=np.array([0,0])):     
 
      """   
      R. G. Steadman, 1979: 
       The Assessment of Sultriness. Part I: A Temperature-Humidity Index Based on Human Physiology and Clothing Science.
        J. Appl. Meteor., 18, 861–873.
        doi: http://dx.doi.org/10.1175/1520-0450(1979)018<0861:TAOSPI>2.0.CO;2 
    
      Lans P. Rothfusz (1990): NWS Technical Attachment (SR 90-23)
    
      The ‘Heat Index’ is a measure of how hot weather "feels" to the body.
      The combination of temperature an humidity produce an "apparent temperature" 
      or the temperature the body "feels". The returned values are for shady locations only. 
      Exposure to full sunshine can increase heat index values by up to 15°F. 
      Also, strong winds, particularly with very hot, dry air, can be extremely 
      hazardous as the wind adds heat to the body
    
      The computation of the heat index is a refinement of a result obtained by multiple 
      regression analysis carried out by Lans P. Rothfusz and described in a 
      1990 National Weather Service (NWS) Technical Attachment (SR 90-23).  
    
      In practice, the Steadman formula is computed first and the result averaged 
      with the temperature. If this heat index value is 80 degrees F or higher, 
      the full regression equation along with any adjustment as described above is applied. 
    
      Note: 
    
      """
      # Default coef are for .ge.80F and 40-100% humidity 
      coef  = np.array([-42.379, 2.04901523, 10.14333127, -0.22475541   \
               ,-0.00683783, -0.05481717, 0.00122874, 0.00085282, -0.00000199 ])
      crit  = np.array([80, 40, 100])    ; # (T_low (F),  RH_low,  RH_High/)
    
      #; Optional coef are for 70F-115F and humidities between 0 and 80% 
      #; Within 3F of default coef
      if opt: 
          coef = np.array([0.363445176, 0.988622465, 4.777114035, -0.114037667  \
                   ,-0.000850208,-0.020716198, 0.000687678,  0.000274954, 0.0 ])
          crit = np.array([ 70, 0, 80])   # F

    
      if iounit[0]==2.:                         # t in degF
          HI = HeatIndexNWS_eqns(t, rh, crit, coef) # use input (t) directly
      else:
          if iounit[0]==0: 
               t = 1.8*t + 32                           # degC => degF
          else:
               t = 1.8*t - 459.67                      # degK => degF
               
          HI = HeatIndexNWS_eqns(t, rh, crit, coef) 

    
      if iounit[1]==0: # output Celsius
              HI = (HI-32)*0.55555
      elif iounit[1]==1:
              HI = (HI+459.67)*0.55555 # output Kelvin
      
      return HI
  
    
def calcH(T,Q,retAll=False):
     
    """
    This function takes temperature and specific humidity and calculates
    the moist enthalpy: the static energy in an air parcel resulting from 
    summing the sensible and latent heat content. 
     
    *Notes*:
     
        - We include a temperature-dependent parameterisation of L (the 
        latent heat of vaporization)
     
        - Cp (the specific heat capacity of air at constant pressure) is 
        parameterised as a function of its mositure content (specific humidity
        - 'spechum'):
         
            Cp=spechum*1952+1005.7*(1-spechum)
         
    Input: 
     
        - T: air temperature (Kelvin)
        - Q: specific humidity (kg/kg)
     
    Ouput: 
     
        - H: Moist enthalpy (J)
     
    """
     
    # Constants
    # Specific heat
    cp_dry=1004.67 # J/Kg/K
     
    # Convert to K if in C
    if np.nanmax(T)<170.: 
         TK=T+273.15
    else: TK=T*1.
    
    # Temp-dependent parameterization of L
    # Original source is Henderson-Sellers (1984); implementation below is 
    # lifted from Davis et al. (2017) - see 10.5194/gmd-10-689-2017
    L= 1.91846 * np.power((TK/(TK-33.91)),2) * np.power(10,6)
     
    # Work out specific heat capacity following Stull, pg  44
    Cp=cp_dry*(1+0.84*(Q/(1-Q)))
     
    Hl=Q*L # Latent heat
    Hs=TK*Cp # Sensible heat
     
    if retAll:
        
        # Moist enthalpy, sensible heat, latent heat, Equivalent temperature
        return Hl+Hs,Hs,Hl# ,(Hl+Hs)/Cp
    else:
        
        return Hl+Hs

def calcDI(press,temp,rh):
    
    """
    This function calculates the 'discomfort index' as provided (by not inveted)
    by Epstein and Moran (2006)
    
    Input: 
        
        -press: air pressure (hPa)
        -tempK: air temperature (Kelvin)
        -rh: relative humidity (fraction 0-->100)
        
    Output: 
        -di: discomfort index
        
    NOTE: no error checking of variables is performed...
        
    """
    # Check for K and convert if in C
    if np.nanmax(temp) <150.:
        tempK=temp+273.15
    else: tempK=temp*1.
    
    tw=np.squeeze(al.calctw(press,tempK,rh))
    #print tw
    di=(0.5*tw+0.5*tempK)-273.15
    
    return di

def calcSWBGT(temp,rh):
    
    """ Computes the simplified wet bulb globe temperature - see Willett 
    and Sherwood, 2010. 
    
    Input: 
        - temp (K or deg C)
        - rh (frac, or 0-100)
        
    Output:
        - swbgt (deg C)
        
    Note: formula wants vapour pressure (e) in hPa. We get this from the satVp
    formula
        
    Formula: W = 0.567T + 0.393e + 3.94
    
    """
    if np.nanmax(rh)>1:
        rh_frac=rh/100.
    else:
        rh_frac=rh*1.
    
    e=satVp(temp)/100.*rh_frac
    
    swbgt=0.57*temp+0.393*e+3.94
    
    return swbgt

def calcATW(tempC,vp,ws):
    
    """Calculates the apparent shade temperature using the algorithm of 
    Steadman (1994).
    
    tempC: input temp in C
    vp: input vapour pressure in hPa
    ws: input wind speed in m/s
    
    returns: AT (degC)
    
    """
    if np.nanmin(tempC) >100.:
        tempC=tempC-273.15
    else: tempC=tempC*1.
    
    if np.nanmin(vp) >100.:
        vp/=100
    else: vp=vp*1.

    at=tempC+0.33*vp-0.7*ws-4.0

    return at    
    
    
def windProfile(ws,ref_z,new_z):
    
    """
    Adjusts wind speed at height ref_z (m) to height new_z, assuming 
    neutral stability
    """
    
    ws_new=ws*np.power((new_z/ref_z),0.143)
    
    return ws_new

def calcWCT(tempC,windKMH):
    
    out=np.zeros(windKMH.shape)*np.nan
    ind=windKMH>5
    out[ind]=13.12+0.6215*tempC[ind]-11.37*np.power(windKMH[ind],0.16) +\
    0.3965*tempC[ind]*np.power(windKMH[ind],0.16)
    
    return out


def dewVp(dewK):
    
    """
    Simple function that calculates vapour pressure given dewpoint temperature
    
    Input:
        
        - dewK: dewpoint temperature in Kelvin
        
        
    Constants:
        
        -Rvl: 0.001844/K
        -T_0:  273.15 K
        -e_0: 610.8 Pa
        
    """
    dewK=np.atleast_1d(dewK)
    Rvl=0.0001844
    T_0=273.15
    e_0=610.8
    vp = e_0*np.exp((1./dewK-1/T_0)/-Rvl)
    return vp


def vpDew(vp):
    
        
    """
    Simple function that calculates dewpoint as a function of vapour pressure

    
    Input:
        
        - dewK: dewpoint temperature in Kelvin
        
        
    Constants:
        
        -Rvl: 0.001844/K
        -T_0:  273.15 K
        -e_0: 610.8 Pa
    """
    vp=np.atleast_1d(vp)
    Rvl=0.0001844
    T_0=273.15
    e_0=610.8
    
    dewK=np.power(-1*np.log(vp/e_0)*Rvl+1/T_0,-1)
    return dewK

 
def satVp(temp):
 
    """ 
    Simply computes the saturation vapour pressure (pascal) for specified temp.
 
    Input: 
        - temp (C or K)
         
    Output:
        - svp: saturation vapour pressure (Pa)
         
    """
    # Check for K and convert if in C
    if np.nanmax(temp) <150.:
        tempK=temp+273.15
    else: tempK=temp*1.
    
    svp=610.8*np.exp(5423*(1/273.15-1/tempK))
    
    return svp
     
def specHum(temp,rh,press):
     
    """ 
    Simply computes the specific humidity (kg/kg) for given temp, relative 
    humidity and air pressure
     
    Input: 
     
        - temp (C or K)
        - rh (fraction): relative humidity
        - press: air pressure (hPa or Pa)
 
    """
     
    # Check for rh as frac and convert if in %
    if np.nanmax(rh)>1.5:
        _rh=rh/100.
    else: _rh = rh*1.    
    # Check for Pa and convert if in hPa (note - no checking for other formats)   
    if np.nanmax(press)<2000:
        _press=press*100.
    else: _press=press*1. 
    # Note that satVp() checks if temp is in K...
    svp=satVp(temp)
    Q=_rh*svp*0.622/_press
    
    return Q

def Q2vp(Q,Press):
    
    """
    This is a very simple function that, given, air pressure, 
    converts specific humidity to vapour pressure. 
    
    Input: 
        
        - Q: specific humidity (Kg/Kg)
        -Press: air pressure (hPa)
        
    Output: 
        - vp: vapour pressure (hPa)
    """
    
    if np.min(Press)>2000:
        Press_hpa=Press/100.
    else: Press_hpa=Press*1.
    
    vp=np.divide(np.multiply(Q,Press_hpa),0.622)
    
    return vp

def calcHumidex(temp,rh):
    
    """
    Calculates the 'humidex' using the equation given in Zhao et al. (2015) 
    
    Input: 
        -temp: air temperature (K or degrees Celsius)
        -rh: relative humidity (frac, 0-->1)
        
    Output:
        -hidx: humidex (degrees Celsius)
        
    NOTE: we use satVp and temp to compute vp from rh
        
    """
    
    if np.nanmax(temp) >70.:
        tempC=temp-273.15
    else: tempC=temp*1.
    if np.nanmax(rh)>1.:
        rh_frac=rh/100.
    else: rh_frac=rh*1.
    
    vp=satVp(temp)*rh_frac/100.
    hidx=tempC+0.555*vp-5.5
    
    return hidx
    
    

def UTCI(ta,mrt,vp,ws):
    
    """ Calls the UTCI approximation routine (long polynomial at bottom of 
    file)
    
    Input:
        
            -Ta       : air temperature, degree Celsius
            -vp       : water vapour presure, hPa
            -mrt      : mean radiant temperature, degree Celsius
            -ws       : wind speed 10 m above ground level in m/s
            
    Output: 
        
            -utci (from f())     : universal thermal climate index, degree celsius
            
    """
    utci=UTCI_eqns(Ta=ta,va=ws,Tmrt=mrt,ehPa=vp)
    return utci
    
      
def Faren2C(f):
    """
    Simply converts array of farenheit to array of Celsius
    """
    const=np.float(5./9.)
    return np.multiply(np.subtract(f,32),const)
      
def geoCart(lat,lon,height,deg=True):
    
    """
    Function takes numpy vectors of latitude, longitude and height (m), and
    uses the WGS84 ellipsoid to compute the cartesian coordinates (x,y,x) (m) 
    
    Note that an m*3 array ("out") is returned 
    """
    lat=np.atleast_1d(lat)
    lon=np.atleast_1d(lon)
    height=np.atleast_1d(height)
    out=np.zeros((len(lat),3))
    if deg:
        lat=np.radians(lat)
        lon=np.radians(lon)
    
    N=wgms84_a/np.sqrt(1-wgms84_e_sq*np.power(np.sin(lat),2))
    x=(N+height)*np.cos(lat)*np.cos(lon)
    y=(N+height)*np.cos(lat)*np.sin(lon)
    z=((1-wgms84_e_sq)*N+height)*np.sin(lat)
    
    out[:,0]=x; out[:,1]=y; out[:,2]=z; return  out # m
    
def interpIDW(ref_points,target_points,data,n=1):
    
    """
    This function interpolates values to target_points, given ref_points and
    data at ref_points. N is the number of neighbours to be used in the 
    interpolation. 
    
    NOTE0: that ref_points and target_points should be ravelled
    arrays (x,y,z)
    
    NOTE1: at present, weights are calculated to be proportional to the 
    square of distance. 
        
    """

    tree=scipy.spatial.cKDTree(ref_points)
    d,inds=tree.query(target_points,k=n); 
    w=1.0/np.power(d,2)
    out=np.sum(w*data[inds],axis=1)/np.sum(w,axis=1)
    
    # need to insert ref_points' exact data at points where distance == zero
    idx=np.min(d,axis=1)==0
    out[idx]=data[inds[idx][:,0]]

    if np.sum(np.isnan(out))>0:
        print "find the nan!"
        return inds[np.isnan(out)],d[np.isnan(out)] 

    
    return out # interpolated values of data at target_points
    
def nearest(x,y,z,refGrid,N):
    
    """
    Function takes x,y,z position and uses scipy's kdtree to locate the 
    N nearest neighbours. Returns the distances and the row/col in a lat 
    (row) and lon (col) grid
    """
    pass
    """ NOT NEEDED!!!"""

def interpIrregHeight(lat_ref,lon_ref,height_ref,lat_i,lon_i,deg=True):
    
    """ 
    Function takes latitudes & longitudes, along with surface
    heights  (numpy VECTORS - ravelled grid), and interpolates height to 
    lat_i,lon_i,height_i. 
    
    NOTE0: intended use is to interpolate height to sample points. More 
    generally, it can be used to interpolate any ellipsoidal-surface field to 
    other points along the WGS1984 ellipsoid's surface.

    """
    
    ref_points=geoCart(lat_ref,lon_ref,np.zeros(len(lat_ref)),deg)
    interp_points=geoCart(lat_i,lon_i,np.zeros(len(lat_i)),deg)
    out=scipy.interpolate.griddata(ref_points,height_ref,interp_points)
    
    return out # heights at requested points
    
def interpIrregSurf(lat_ref,lon_ref,height_ref,data,lat_i,lon_i,height_i,deg=True):
    
    """ 
    Function takes latitudes & longitudes, along with surface
    heights & data (numpy VECTORS - ravelled grid), and interpolates to 
    lat_i,lon_i,height_i
    
    NOTE0: intended use is to interpolate surface meteorological fields to 
    discrete points on the Earth's surface - which can be above/below the 
    ellipsoidal surface. 
    
    NOTE1: works by first converting lat_ref,lon_ref,height_ref to cartesian
    coordinates, and then calling numpy.griddata to interpolate to these 
    positions. Linear interpolation is used here as the default. 
    
    """
    ref_points=geoCart(lat_ref,lon_ref,height_ref,deg)
    interp_points=geoCart(lat_i,lon_i,height_i,deg)
    out=scipy.interpolate.griddata(ref_points,data,interp_points)
    
    return out # met surface fields at the queried lat/lon/height
    
def muPos(lons1,lats1,lons2,lats2,degrees=True): 
    
    """
    Function takes pairs of vectors (lons1/2 and lats/1/2) and then finds the mean
    position for each pair of coordinates. This is equivalent to finding the
    middle points along great circle arcs. 
    
    NB. This function relies on the third-party package 'nvector'
    NB. Returns lat/lon
    """
    
    if degrees:
        lons1=np.radians(lons1)
        lats1=np.radians(lats1)
        lons2=np.radians(lons2)
        lats2=np.radians(lats2)        
    
    # convert lats/lons to e-vectors
    ev1=nv.lat_lon2n_E(lats1,lons1) # 3*n
    ev2=nv.lat_lon2n_E(lats2,lons2)
    
    # Iterate over pairs of nvectors (ev1/ev2) (3*npoints) 
    # and get mean position for each pair
    
    return np.squeeze(np.array([\
    np.degrees(nv._core.n_E2lat_lon(nv._core.mean_horizontal_position\
    (np.column_stack((ev1[:,ii],ev2[:,ii]))))) for ii in range(len(lats1))]))
    
def lon_lat_to_cartesian(lon, lat,unit=True):
    
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R (defined at top of script)
    """
    
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    if unit: # unit sphere
        R=1
    else: 
        R=6378.1370
    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    
    return x,y,z
    
def cartesian_to_lon_lat(x,y,z):
    
    """ self explanatory"""
    
    lat=np.degrees(np.arcsin(z))
    if x>0:
        lon=np.degrees(np.arctan(y/x))
    elif y>0:
        lon=np.degrees(np.arctan(y/x))+180
    else:
        lon=np.degrees(np.arctan(y/x))-180
        
    return lon,lat

def block_length_AR1(series):
    
    """
    Computes the optimum block length for block bootstrap.Solves the implicit 
    equation: 
        
        L=(n-L+1)^(2/3)*(1-n'/n)    (Wilks, 2005, p.178)
        note: n = (1-p1)/(1+p1)     (Wilks, 2005 p.147)
        and p1 is the lag-1 autocorrelation coefficient
        
    The equation must be solved iteratively; this function uses scipy's 
    optimization power
    """
    def eq(L):
        """ should be initialized by sqrt(n)"""
        error=np.abs(L-(n-L+1)**(1./3.*(1-nprime/n)))
        return error
    
    n=len(series)
    p1=np.corrcoef(series[1:],series[:-1])[0,1]
    nprime=(1-p1)/(1+p1)
    L=scipy.optimize.minimize(eq,np.sqrt(np.float(n)))
    
    return np.int(np.floor(L.x))

def block_bootstrap_trend(series,niter):
    
    """ 
    Employs a block bootstrap to build niter series; evaluates theil-sen slope
    on this. Returns the 95% confidence limits for the null distribution
    """
    
    L=block_length_AR1(series)
    out=np.zeros((niter))
    n=len(series)
    nb=np.int(n/np.float(L))
    for ii in range(niter):
        samp=np.zeros(n)
        count=0
        for jj in range(nb):
            st_pos=randint(0,n-L)
            block=series[st_pos:st_pos+L]
            samp[count:count+L]=block
            count+=L
        out[ii]=stats.mstats.theilslopes(samp)[0]
    theil=stats.mstats.theilslopes(series)
    trend=theil[0]; intercept=theil[1]
    p=np.sum(np.abs(out)>=np.abs(trend))/np.float(niter)
    return out,trend,intercept,p,L
        

def theilslopes(y,x=None):
    
    r"""
    Computes the Theil-Sen estimator for a set of points (x, y).
    `theilslopes` implements a method for robust linear regression. It
    computes the slope as the median of all slopes between paired values.
    Parameters
    ----------
    y : array_like
    Dependent variable.
    x : array_like or None, optional
    Independent variable. If None, use ``arange(len(y))`` instead.
    alpha : float, optional
    Confidence degree between 0 and 1. Default is 95% confidence.
    Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are
    interpreted as "find the 90% confidence interval".
    Returns
    -------
    medslope : float
    Theil slope.
    slope std err: float
    -----
    The implementation of `theilslopes` follows [1]

    References
    ----------
    .. [1] P.K. Sen, "Estimates of the regression coefficient based on Kendall's tau",
    J. Am. Stat. Assoc., Vol. 63, pp. 1379-1389, 1968.
    .. [2] H. Theil, "A rank-invariant method of linear and polynomial
    regression analysis I, II and III", Nederl. Akad. Wetensch., Proc.
    53:, pp. 386-392, pp. 521-525, pp. 1397-1412, 1950.
    .. [3] W.L. Conover, "Practical nonparametric statistics", 2nd ed.,
    John Wiley and Sons, New York, pp. 493.

    """
    # We copy both x and y so we can use _find_repeats.
    y = np.array(y).flatten()
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.array(x, dtype=float).flatten()
    if len(x) != len(y):
        raise ValueError("Incompatible lengths ! (%s<>%s)" % (len(y), len(x)))
    # Compute sorted slopes only when deltax > 0
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    slopes.sort()
    medslope = np.median(slopes)

    return medslope, np.std(slopes)/np.sqrt(len(slopes))

def UTCI_eqns(Ta,va,Tmrt,ehPa):
      D_Tmrt=Tmrt-Ta
      Pa = ehPa/10.0
      return Ta+\
		( 6.07562052E-01 )   + \
		( -2.27712343E-02 ) * Ta + \
		( 8.06470249E-04 ) * Ta*Ta + \
		( -1.54271372E-04 ) * Ta*Ta*Ta + \
		( -3.24651735E-06 ) * Ta*Ta*Ta*Ta + \
		( 7.32602852E-08 ) * Ta*Ta*Ta*Ta*Ta + \
		( 1.35959073E-09 ) * Ta*Ta*Ta*Ta*Ta*Ta + \
		( -2.25836520E+00 ) * va + \
		( 8.80326035E-02 ) * Ta*va + \
		( 2.16844454E-03 ) * Ta*Ta*va + \
		( -1.53347087E-05 ) * Ta*Ta*Ta*va + \
		( -5.72983704E-07 ) * Ta*Ta*Ta*Ta*va + \
		( -2.55090145E-09 ) * Ta*Ta*Ta*Ta*Ta*va + \
		( -7.51269505E-01 ) * va*va + \
		( -4.08350271E-03 ) * Ta*va*va + \
		( -5.21670675E-05 ) * Ta*Ta*va*va + \
		( 1.94544667E-06 ) * Ta*Ta*Ta*va*va + \
		( 1.14099531E-08 ) * Ta*Ta*Ta*Ta*va*va + \
		( 1.58137256E-01 ) * va*va*va + \
		( -6.57263143E-05 ) * Ta*va*va*va + \
		( 2.22697524E-07 ) * Ta*Ta*va*va*va + \
		( -4.16117031E-08 ) * Ta*Ta*Ta*va*va*va + \
		( -1.27762753E-02 ) * va*va*va*va + \
		( 9.66891875E-06 ) * Ta*va*va*va*va + \
		( 2.52785852E-09 ) * Ta*Ta*va*va*va*va + \
		( 4.56306672E-04 ) * va*va*va*va*va + \
		( -1.74202546E-07 ) * Ta*va*va*va*va*va + \
		( -5.91491269E-06 ) * va*va*va*va*va*va + \
		( 3.98374029E-01 ) * D_Tmrt + \
		( 1.83945314E-04 ) * Ta*D_Tmrt + \
		( -1.73754510E-04 ) * Ta*Ta*D_Tmrt + \
		( -7.60781159E-07 ) * Ta*Ta*Ta*D_Tmrt + \
		( 3.77830287E-08 ) * Ta*Ta*Ta*Ta*D_Tmrt + \
		( 5.43079673E-10 ) * Ta*Ta*Ta*Ta*Ta*D_Tmrt + \
		( -2.00518269E-02 ) * va*D_Tmrt + \
		( 8.92859837E-04 ) * Ta*va*D_Tmrt + \
		( 3.45433048E-06 ) * Ta*Ta*va*D_Tmrt + \
		( -3.77925774E-07 ) * Ta*Ta*Ta*va*D_Tmrt + \
		( -1.69699377E-09 ) * Ta*Ta*Ta*Ta*va*D_Tmrt + \
		( 1.69992415E-04 ) * va*va*D_Tmrt + \
		( -4.99204314E-05 ) * Ta*va*va*D_Tmrt + \
		( 2.47417178E-07 ) * Ta*Ta*va*va*D_Tmrt + \
		( 1.07596466E-08 ) * Ta*Ta*Ta*va*va*D_Tmrt + \
		( 8.49242932E-05 ) * va*va*va*D_Tmrt + \
		( 1.35191328E-06 ) * Ta*va*va*va*D_Tmrt + \
		( -6.21531254E-09 ) * Ta*Ta*va*va*va*D_Tmrt + \
		( -4.99410301E-06 ) * va*va*va*va*D_Tmrt + \
		( -1.89489258E-08 ) * Ta*va*va*va*va*D_Tmrt + \
		( 8.15300114E-08 ) * va*va*va*va*va*D_Tmrt + \
		( 7.55043090E-04 ) * D_Tmrt*D_Tmrt + \
		( -5.65095215E-05 ) * Ta*D_Tmrt*D_Tmrt + \
		( -4.52166564E-07 ) * Ta*Ta*D_Tmrt*D_Tmrt + \
		( 2.46688878E-08 ) * Ta*Ta*Ta*D_Tmrt*D_Tmrt + \
		( 2.42674348E-10 ) * Ta*Ta*Ta*Ta*D_Tmrt*D_Tmrt + \
		( 1.54547250E-04 ) * va*D_Tmrt*D_Tmrt + \
		( 5.24110970E-06 ) * Ta*va*D_Tmrt*D_Tmrt + \
		( -8.75874982E-08 ) * Ta*Ta*va*D_Tmrt*D_Tmrt + \
		( -1.50743064E-09 ) * Ta*Ta*Ta*va*D_Tmrt*D_Tmrt + \
		( -1.56236307E-05 ) * va*va*D_Tmrt*D_Tmrt + \
		( -1.33895614E-07 ) * Ta*va*va*D_Tmrt*D_Tmrt + \
		( 2.49709824E-09 ) * Ta*Ta*va*va*D_Tmrt*D_Tmrt + \
		( 6.51711721E-07 ) * va*va*va*D_Tmrt*D_Tmrt + \
		( 1.94960053E-09 ) * Ta*va*va*va*D_Tmrt*D_Tmrt + \
		( -1.00361113E-08 ) * va*va*va*va*D_Tmrt*D_Tmrt + \
		( -1.21206673E-05 ) * D_Tmrt*D_Tmrt*D_Tmrt + \
		( -2.18203660E-07 ) * Ta*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 7.51269482E-09 ) * Ta*Ta*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 9.79063848E-11 ) * Ta*Ta*Ta*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 1.25006734E-06 ) * va*D_Tmrt*D_Tmrt*D_Tmrt + \
		( -1.81584736E-09 ) * Ta*va*D_Tmrt*D_Tmrt*D_Tmrt + \
		( -3.52197671E-10 ) * Ta*Ta*va*D_Tmrt*D_Tmrt*D_Tmrt + \
		( -3.36514630E-08 ) * va*va*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 1.35908359E-10 ) * Ta*va*va*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 4.17032620E-10 ) * va*va*va*D_Tmrt*D_Tmrt*D_Tmrt + \
		( -1.30369025E-09 ) * D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 4.13908461E-10 ) * Ta*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 9.22652254E-12 ) * Ta*Ta*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( -5.08220384E-09 ) * va*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( -2.24730961E-11 ) * Ta*va*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 1.17139133E-10 ) * va*va*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 6.62154879E-10 ) * D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 4.03863260E-13 ) * Ta*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 1.95087203E-12 ) * va*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( -4.73602469E-12 ) * D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt + \
		( 5.12733497E+00 ) * Pa + \
		( -3.12788561E-01 ) * Ta*Pa + \
		( -1.96701861E-02 ) * Ta*Ta*Pa + \
		( 9.99690870E-04 ) * Ta*Ta*Ta*Pa + \
		( 9.51738512E-06 ) * Ta*Ta*Ta*Ta*Pa + \
		( -4.66426341E-07 ) * Ta*Ta*Ta*Ta*Ta*Pa + \
		( 5.48050612E-01 ) * va*Pa + \
		( -3.30552823E-03 ) * Ta*va*Pa + \
		( -1.64119440E-03 ) * Ta*Ta*va*Pa + \
		( -5.16670694E-06 ) * Ta*Ta*Ta*va*Pa + \
		( 9.52692432E-07 ) * Ta*Ta*Ta*Ta*va*Pa + \
		( -4.29223622E-02 ) * va*va*Pa + \
		( 5.00845667E-03 ) * Ta*va*va*Pa + \
		( 1.00601257E-06 ) * Ta*Ta*va*va*Pa + \
		( -1.81748644E-06 ) * Ta*Ta*Ta*va*va*Pa + \
		( -1.25813502E-03 ) * va*va*va*Pa + \
		( -1.79330391E-04 ) * Ta*va*va*va*Pa + \
		( 2.34994441E-06 ) * Ta*Ta*va*va*va*Pa + \
		( 1.29735808E-04 ) * va*va*va*va*Pa + \
		( 1.29064870E-06 ) * Ta*va*va*va*va*Pa + \
		( -2.28558686E-06 ) * va*va*va*va*va*Pa + \
		( -3.69476348E-02 ) * D_Tmrt*Pa + \
		( 1.62325322E-03 ) * Ta*D_Tmrt*Pa + \
		( -3.14279680E-05 ) * Ta*Ta*D_Tmrt*Pa + \
		( 2.59835559E-06 ) * Ta*Ta*Ta*D_Tmrt*Pa + \
		( -4.77136523E-08 ) * Ta*Ta*Ta*Ta*D_Tmrt*Pa + \
		( 8.64203390E-03 ) * va*D_Tmrt*Pa + \
		( -6.87405181E-04 ) * Ta*va*D_Tmrt*Pa + \
		( -9.13863872E-06 ) * Ta*Ta*va*D_Tmrt*Pa + \
		( 5.15916806E-07 ) * Ta*Ta*Ta*va*D_Tmrt*Pa + \
		( -3.59217476E-05 ) * va*va*D_Tmrt*Pa + \
		( 3.28696511E-05 ) * Ta*va*va*D_Tmrt*Pa + \
		( -7.10542454E-07 ) * Ta*Ta*va*va*D_Tmrt*Pa + \
		( -1.24382300E-05 ) * va*va*va*D_Tmrt*Pa + \
		( -7.38584400E-09 ) * Ta*va*va*va*D_Tmrt*Pa + \
		( 2.20609296E-07 ) * va*va*va*va*D_Tmrt*Pa + \
		( -7.32469180E-04 ) * D_Tmrt*D_Tmrt*Pa + \
		( -1.87381964E-05 ) * Ta*D_Tmrt*D_Tmrt*Pa + \
		( 4.80925239E-06 ) * Ta*Ta*D_Tmrt*D_Tmrt*Pa + \
		( -8.75492040E-08 ) * Ta*Ta*Ta*D_Tmrt*D_Tmrt*Pa + \
		( 2.77862930E-05 ) * va*D_Tmrt*D_Tmrt*Pa + \
		( -5.06004592E-06 ) * Ta*va*D_Tmrt*D_Tmrt*Pa + \
		( 1.14325367E-07 ) * Ta*Ta*va*D_Tmrt*D_Tmrt*Pa + \
		( 2.53016723E-06 ) * va*va*D_Tmrt*D_Tmrt*Pa + \
		( -1.72857035E-08 ) * Ta*va*va*D_Tmrt*D_Tmrt*Pa + \
		( -3.95079398E-08 ) * va*va*va*D_Tmrt*D_Tmrt*Pa + \
		( -3.59413173E-07 ) * D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( 7.04388046E-07 ) * Ta*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( -1.89309167E-08 ) * Ta*Ta*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( -4.79768731E-07 ) * va*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( 7.96079978E-09 ) * Ta*va*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( 1.62897058E-09 ) * va*va*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( 3.94367674E-08 ) * D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( -1.18566247E-09 ) * Ta*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( 3.34678041E-10 ) * va*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( -1.15606447E-10 ) * D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*Pa + \
		( -2.80626406E+00 ) * Pa*Pa + \
		( 5.48712484E-01 ) * Ta*Pa*Pa + \
		( -3.99428410E-03 ) * Ta*Ta*Pa*Pa + \
		( -9.54009191E-04 ) * Ta*Ta*Ta*Pa*Pa + \
		( 1.93090978E-05 ) * Ta*Ta*Ta*Ta*Pa*Pa + \
		( -3.08806365E-01 ) * va*Pa*Pa + \
		( 1.16952364E-02 ) * Ta*va*Pa*Pa + \
		( 4.95271903E-04 ) * Ta*Ta*va*Pa*Pa + \
		( -1.90710882E-05 ) * Ta*Ta*Ta*va*Pa*Pa + \
		( 2.10787756E-03 ) * va*va*Pa*Pa + \
		( -6.98445738E-04 ) * Ta*va*va*Pa*Pa + \
		( 2.30109073E-05 ) * Ta*Ta*va*va*Pa*Pa + \
		( 4.17856590E-04 ) * va*va*va*Pa*Pa + \
		( -1.27043871E-05 ) * Ta*va*va*va*Pa*Pa + \
		( -3.04620472E-06 ) * va*va*va*va*Pa*Pa + \
		( 5.14507424E-02 ) * D_Tmrt*Pa*Pa + \
		( -4.32510997E-03 ) * Ta*D_Tmrt*Pa*Pa + \
		( 8.99281156E-05 ) * Ta*Ta*D_Tmrt*Pa*Pa + \
		( -7.14663943E-07 ) * Ta*Ta*Ta*D_Tmrt*Pa*Pa + \
		( -2.66016305E-04 ) * va*D_Tmrt*Pa*Pa + \
		( 2.63789586E-04 ) * Ta*va*D_Tmrt*Pa*Pa + \
		( -7.01199003E-06 ) * Ta*Ta*va*D_Tmrt*Pa*Pa + \
		( -1.06823306E-04 ) * va*va*D_Tmrt*Pa*Pa + \
		( 3.61341136E-06 ) * Ta*va*va*D_Tmrt*Pa*Pa + \
		( 2.29748967E-07 ) * va*va*va*D_Tmrt*Pa*Pa + \
		( 3.04788893E-04 ) * D_Tmrt*D_Tmrt*Pa*Pa + \
		( -6.42070836E-05 ) * Ta*D_Tmrt*D_Tmrt*Pa*Pa + \
		( 1.16257971E-06 ) * Ta*Ta*D_Tmrt*D_Tmrt*Pa*Pa + \
		( 7.68023384E-06 ) * va*D_Tmrt*D_Tmrt*Pa*Pa + \
		( -5.47446896E-07 ) * Ta*va*D_Tmrt*D_Tmrt*Pa*Pa + \
		( -3.59937910E-08 ) * va*va*D_Tmrt*D_Tmrt*Pa*Pa + \
		( -4.36497725E-06 ) * D_Tmrt*D_Tmrt*D_Tmrt*Pa*Pa + \
		( 1.68737969E-07 ) * Ta*D_Tmrt*D_Tmrt*D_Tmrt*Pa*Pa + \
		( 2.67489271E-08 ) * va*D_Tmrt*D_Tmrt*D_Tmrt*Pa*Pa + \
		( 3.23926897E-09 ) * D_Tmrt*D_Tmrt*D_Tmrt*D_Tmrt*Pa*Pa + \
		( -3.53874123E-02 ) * Pa*Pa*Pa + \
		( -2.21201190E-01 ) * Ta*Pa*Pa*Pa + \
		( 1.55126038E-02 ) * Ta*Ta*Pa*Pa*Pa + \
		( -2.63917279E-04 ) * Ta*Ta*Ta*Pa*Pa*Pa + \
		( 4.53433455E-02 ) * va*Pa*Pa*Pa + \
		( -4.32943862E-03 ) * Ta*va*Pa*Pa*Pa + \
		( 1.45389826E-04 ) * Ta*Ta*va*Pa*Pa*Pa + \
		( 2.17508610E-04 ) * va*va*Pa*Pa*Pa + \
		( -6.66724702E-05 ) * Ta*va*va*Pa*Pa*Pa + \
		( 3.33217140E-05 ) * va*va*va*Pa*Pa*Pa + \
		( -2.26921615E-03 ) * D_Tmrt*Pa*Pa*Pa + \
		( 3.80261982E-04 ) * Ta*D_Tmrt*Pa*Pa*Pa + \
		( -5.45314314E-09 ) * Ta*Ta*D_Tmrt*Pa*Pa*Pa + \
		( -7.96355448E-04 ) * va*D_Tmrt*Pa*Pa*Pa + \
		( 2.53458034E-05 ) * Ta*va*D_Tmrt*Pa*Pa*Pa + \
		( -6.31223658E-06 ) * va*va*D_Tmrt*Pa*Pa*Pa + \
		( 3.02122035E-04 ) * D_Tmrt*D_Tmrt*Pa*Pa*Pa + \
		( -4.77403547E-06 ) * Ta*D_Tmrt*D_Tmrt*Pa*Pa*Pa + \
		( 1.73825715E-06 ) * va*D_Tmrt*D_Tmrt*Pa*Pa*Pa + \
		( -4.09087898E-07 ) * D_Tmrt*D_Tmrt*D_Tmrt*Pa*Pa*Pa + \
		( 6.14155345E-01 ) * Pa*Pa*Pa*Pa + \
		( -6.16755931E-02 ) * Ta*Pa*Pa*Pa*Pa + \
		( 1.33374846E-03 ) * Ta*Ta*Pa*Pa*Pa*Pa + \
		( 3.55375387E-03 ) * va*Pa*Pa*Pa*Pa + \
		( -5.13027851E-04 ) * Ta*va*Pa*Pa*Pa*Pa + \
		( 1.02449757E-04 ) * va*va*Pa*Pa*Pa*Pa + \
		( -1.48526421E-03 ) * D_Tmrt*Pa*Pa*Pa*Pa + \
		( -4.11469183E-05 ) * Ta*D_Tmrt*Pa*Pa*Pa*Pa + \
		( -6.80434415E-06 ) * va*D_Tmrt*Pa*Pa*Pa*Pa + \
		( -9.77675906E-06 ) * D_Tmrt*D_Tmrt*Pa*Pa*Pa*Pa + \
		( 8.82773108E-02 ) * Pa*Pa*Pa*Pa*Pa + \
		( -3.01859306E-03 ) * Ta*Pa*Pa*Pa*Pa*Pa + \
		( 1.04452989E-03 ) * va*Pa*Pa*Pa*Pa*Pa + \
		( 2.47090539E-04 ) * D_Tmrt*Pa*Pa*Pa*Pa*Pa + \
		( 1.48348065E-03 ) * Pa*Pa*Pa*Pa*Pa*Pa 
