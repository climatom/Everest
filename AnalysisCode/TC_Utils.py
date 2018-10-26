#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Cyclones function file. 

Here we store the main methods used to query hurricane/heatwave frequency 
probability. 
"""

# Import modules
import numpy as np, os, datetime, gzip, itertools, GeneralFunctions as GF, pandas as pd
from netCDF4 import Dataset
from numba import jit
import sys, tarfile
#from cdo import*
#cdo=Cdo()

# Define Functions

def get_rc(lons,lats,times,grid_lon,grid_lat):
    
    """
    
    This fuction takes the lon/lat/time (dec day) of TC and finds the 
    srow/col of the heatwave grid
    
    
    """
    
    n=len(lats)
    
    # Prescribe index arrays from 1:nlat and 1:nlon
    la_idx=np.arange(len(grid_lat))
    lo_idx=np.arange(len(grid_lon))
    
    # Now find closest lon/lats              
    rows=[la_idx[np.argmin(np.abs(lats[ii]-grid_lat))] for ii in range(n)]
    cols=[lo_idx[np.argmin(np.abs(lons[ii]-grid_lon))] for ii in range(n)]
     
    
    return rows,cols

@jit
def get_rc_mask(lons,lats,grid_lon,grid_lat,mask):
    
    """
    
    Same as above but also takes a mask as input. If the closest grid cell
    is masked, the next closest is taken, and so-forth. 
    
    
    """

    n=len(lats)
    out_rows=np.zeros(n,dtype=np.int32)
    out_cols=np.zeros(n,dtype=np.int32)
    
    # Prescribe index arrays from 1:nlat and 1:nlon
    la_idx=np.arange(len(grid_lat))
    lo_idx=np.arange(len(grid_lon))
    
    # loop over input coords
    nmsk=0
    for ii in range(n):        
        la_idx_sort=la_idx[np.argsort(np.abs(lats[ii]-grid_lat))]
        lo_idx_sort=lo_idx[np.argsort(np.abs(lons[ii]-grid_lon))]
        
        jj=0
        while mask[la_idx_sort[jj],lo_idx_sort[jj]]:
            nmsk+=1
            jj+=1
            
        out_rows[ii]=la_idx_sort[jj]; out_cols[ii]=lo_idx_sort[jj]
        
    return out_rows,out_cols ,nmsk


@jit
def get_rc_mask_dist(lons,lats,grid_lon,grid_lat,mask):
    
    """
    
    Same as above but uses great-circle distances to compute the 
    nearest neighbours. Note that it also returns the distance between 
    the grid cell selected, and the TC landfall location
    
    Returns:
	- TC lat
	- TC lon 
	- Grid lat
	- Grid lon
	- Grid row
	- Grid col
	- Distance (km) to TC landfall location

    """

    # Query the input dimensions
    n=len(lats)
    nrows,ncols=grid_lon.shape

    # Preallocate array array for output (see preamble for vars out)
    out=np.zeros((n,8))*np.nan

    # Create an index grid the same shape as the grid_lon/lat arrays 
    col_grid,row_grid=np.meshgrid(range(ncols),range(nrows))

    # Flatten
    col_grid=col_grid.flatten(); row_grid=row_grid.flatten()
    grid_lon=grid_lon.flatten(); grid_lat=grid_lat.flatten()
    mask=mask.flatten()

    # Create a reference vector from 1:nelem (enables easy way to keep track of 
    # which locations have been selected)
    ref=np.arange(len(grid_lon))
	
    # Iterate over the input lon/lats
    for ii in range(n):
	
    	# Feed to the [fast-ish] Haversine function
    	d=GF.haversine_fast(lats[ii],lons[ii],grid_lat,grid_lon,miles=False)
   

	# Order from low [0] --> high [nelem]
        order=np.argsort(d)

        # This loop will continue selecting grid cells further and further away 
        # until we have a lon/lat that is also classified as land in the grid data
	jj=0
	while mask[order][jj]:
		jj+=1

	out[ii,0]=lats[ii]; out[ii,1]=lons[ii]
	out[ii,2]=grid_lat[order][jj]; out[ii,3]=grid_lon[order][jj]
	out[ii,4]=row_grid[order][jj]; out[ii,5]=col_grid[order][jj]
        out[ii,6]=d[order][jj]; out[ii,7]=ref[order][jj]
	        

    return out,out[:,4],out[:,5]


              
@jit
def get_rc_time_mask(lons,lats,times,grid_lon,grid_lat,grid_times,mask,\
                     time_thresh):
       
    """
    
    Same as above but also takes the right time. 
    
    """
   
    # Preallocate 
    n=len(lats)
    out_rows=np.zeros(n,dtype=np.int32)
    out_cols=np.zeros(n,dtype=np.int32)
    out_times=np.zeros((n,time_thresh),dtype=np.int32)*np.nan
    
    # Prescribe index arrays from 1:nlat and 1:nlon
    la_idx=np.arange(len(grid_lat))
    lo_idx=np.arange(len(grid_lon))
    t_idx=np.arange(len(grid_times))
    
    # Loop over input coords
    nmsk=0
    for ii in range(n):
        time_crit=np.floor(times[ii])
        la_idx_sort=la_idx[np.argsort(np.abs(lats[ii]-grid_lat))]
        lo_idx_sort=lo_idx[np.argsort(np.abs(lons[ii]-grid_lon))]
        t_idx_ii=np.logical_and(grid_times>=time_crit+1,grid_times<=\
                                (time_crit+time_thresh+1))

        # The time index is NOT sensitive to missing values
        out_times[ii,:np.sum(t_idx_ii)]=t_idx[t_idx_ii]; \

        
        # Keep checking nearest neighbours until we get a hit (non-missing)
        jj=0
        while mask[la_idx_sort[jj],lo_idx_sort[jj]]:
            nmsk+=1
            jj+=1
            
        out_rows[ii]=la_idx_sort[jj]
        out_cols[ii]=lo_idx_sort[jj]

    return out_rows,out_cols,out_times
               

def get_ll_rc_time(lons,lats,times,grid_lon,grid_lat,grid_times,time_thresh):
    
    """
    
    This fuction takes the lon/lat/time (dec day) of TC and finds the 
    stack/row/col/ of the heatwave grid
    
    
    """
    
    # number of TCs we have 
    n=len(lats)
    
    # Prescribe index arrays from 1:nlat and 1:nlon
    la_idx=np.arange(len(grid_lat))
    lo_idx=np.arange(len(grid_lon))
    t_idx=np.arange(len(grid_times))
    
    # Now find closest lon/lats              
    rows=[la_idx[np.argmin(np.abs(lats[ii]-grid_lat))] for ii in range(n)]
    cols=[lo_idx[np.argmin(np.abs(lons[ii]-grid_lon))] for ii in range(n)]
    
    # Indices for time masking    
    stacks=[t_idx[np.logical_and\
     (grid_times>=times[ii]+1,grid_times<=times[ii]+time_thresh)] \
     for ii in range(n)]    
    
    return rows,cols,stacks

def points_within_km(target_lon,target_lat,ref_lon,ref_lat,thresh):
    
    """
    Designed to take a target point and find all those locations within
    thresh km from it
    
    Returs array [ref_lon,ref_lat] where dist is below thresh
    """
    
    dists=GF.haversine_fast(target_lat,target_lon,ref_lat,ref_lon,miles=False)
    ind=dists<thresh
    return np.column_stack((ref_lon[ind],ref_lat[ind]))

def points_within_km_copyMeta(target_lon,target_lat,target_time,target_id,\
                              ref_lon,ref_lat,thresh):
    
    """
    Same as above, but we have added the time variable ad TC id; this is simply added
    copied/exaded to have the same length/dimensions as the number of 
    successful candiadate distances
    
    Returs array [ref_lon,ref_lat] where dist is below thresh
    """
    
    dists=GF.haversine_fast(target_lat,target_lon,ref_lat,ref_lon,miles=False)
    ind=dists<thresh
    time_out=np.tile(target_time,np.sum(ind))
    id_out=np.tile(target_id,np.sum(ind))
    return np.column_stack((ref_lon[ind],ref_lat[ind],time_out,id_out))

#def interpNN(lon,lat,gridfile,var):
    
    """ 
    This function uses CDO python bindings to interpolate to a point of 
    interest. It returns a 1-d time series.
    
    Notes:
        - lon/lat = coordinate to interpolate to 
        - gridfile = netcdf file to interpolate
        - var = string with name of variable to interpolate
    
    """
    
    #out=cdo.remapnn("lon=%.3f_lat=%.3f"%(lon,lat),input=gridfile,returnArray=var)
    #return out
    
def seasonal_cycle(array,dec_time):

    pass

#@jit
#def get_hi(lons,lats,times,grid_lon,grid_lat,grid_times,time_thresh,hi):
#    
#    """
#    
#    This function calls get_ll_rc_time and iterates over the indices, 
#    extracting the hi data as it goes
#    n

#    """
#
#    rows,cols,stacks=get_ll_rc_time\
#    (lons,lats,times,grid_lon,grid_lat,grid_times,time_thresh)
#    n=len(rows)
#    nt=len(stacks[0])+1
#    hi_out=np.zeros((n,nt))*np.nan
#
#    for ii in range(n):
#        hi_out[ii,:len(stacks[ii])]=hi[stacks[ii],rows[ii],cols[ii]]
#
#        
#    return hi_out
 
@jit
def get_hi_mask(lons,lats,times,grid_lon,grid_lat,grid_times,\
           mask,time_thresh,hi):
    
    """
    
    This function calls get_rc_time_mask and iterates over the indices, 
    extracting the hi data as it goes. It refu
                    
    """

    # Get the rows/cols/times ("stack" index) for all the TCs - being aware of
    # missing values
    rows,cols,stacks=get_rc_time_mask\
    (lons,lats,times,grid_lon,grid_lat,grid_times,mask,time_thresh)
    
    # Now loop over all the rows/cols and take the corresponding times
    n=len(rows)
    hi_out=np.zeros((n,time_thresh))*np.nan

    # In this loop we iterate over the TC time steps and extract data for the
    # correct time/row/col
    for ii in range(n):
        ntime=np.int(np.sum(~np.isnan(stacks[ii,:])))
        t_idx=(stacks[ii,:ntime]).astype(np.int)
        hi_out[ii,:ntime]=hi[t_idx,rows[ii],cols[ii]]
        
        
        
    return hi_out    

@jit
def get_hi_mask_smart(lons,lats,times,grid_lon,grid_lat,grid_times,\
           mask,time_thresh,hi,next_file,vname):
    
    """
    
    This function calls get_ll_rc_time and iterates over the indices, 
    extracting the hi data as it goes. It is "smart" to TCs occuring 
    at the end of the year - and therefore requring next year's data to 
    correctly identify all heat index values falling within the required 
    time interval of the event ("time_thresh"). Function is otherwise the
    same as "get_hi_mask", above.
    
    NOTES: 
        
        -"next_file" is the name of the next yearly HI file. It will be 
        opened and the first k days will be read if the number of days
        taken from the current file is less than timw_thresh and the TC_time 
        is >335. 
        
        -"vname" is the name of the variable (string) in the netCDF file that
        we want to extract in next_file. 
            
    
    """

    # Get the rows/cols/times ("stack" index) for all the TCs - being aware of
    # missing values
    rows,cols,stacks=get_rc_time_mask\
    (lons,lats,times,grid_lon,grid_lat,grid_times,mask,time_thresh)
    
    # Now loop over all the rows/cols and take the corresponding times
    n=len(rows)
    hi_out=np.zeros((n,time_thresh))*np.nan

    # In this loop we iterate over the TC time steps 
    for ii in range(n):
        ntime=np.int(np.sum(~np.isnan(stacks[ii,:])))
        # Guard against NO time steps 
        if ntime >0:
            t_idx=(stacks[ii,:ntime]).astype(np.int)
            hi_out[ii,:ntime]=hi[t_idx,rows[ii],cols[ii]]
        
        # Check for a time slice being <time_thresh long. If that happens, 
        # it means the TC occurred at the end of the year, and we should 
        # therefore take the next yearly file, extracting the first n-k 
        # elements - where n=time_thresh and k=the number of time steps we 
        # actually have        
        if ntime <(time_thresh-1):
            #print "Opening next file. DoY = %.0f; ntime=%.0f" % \
#            (np.min(grid_times[t_idx]),ntime)
            df=Dataset(next_file,"r")
            hi_out[ii,ntime:]=\
            df[vname][:(time_thresh-ntime),rows[ii],cols[ii]]
        
    return hi_out


@jit
def get_hi_locs_smart(lons,lats,times,grid_lon,grid_lat,grid_times,\
           mask,time_thresh,heat_thresh,hi,next_file,vname):
    
    """
    
    This is identical to the above, but it also returns  grid of 1/0 in cells
    recording "hits" - that is, heat index values >= heat_thresh
    
    """

    # Get the rows/cols/times ("stack" index) for all the TCs - being aware of
    # missing values
    rows,cols,stacks=get_rc_time_mask\
    (lons,lats,times,grid_lon,grid_lat,grid_times,mask,time_thresh)
    
    # Now loop over all the rows/cols and take the corresponding times
    n=len(rows)
    hi_out=np.zeros((n,time_thresh))*np.nan
    
    # Preallocate "hit" grid - two layers of zeros 
    hit_grid=np.zeros((len(grid_lat),len(grid_lon)))

    # In this loop we iterate over the TC time steps and 
    for ii in range(n):
        ntime=np.int(np.sum(~np.isnan(stacks[ii,:])))
        t_idx=(stacks[ii,:ntime]).astype(np.int)
        hi_out[ii,:ntime]=hi[t_idx,rows[ii],cols[ii]]
        
        # Check for a time slice being <time_thresh long. If that happens, 
        # it means the TC occurred at the end of the year, and we should 
        # therefore take the next yearly file, extracting the first n-k 
        # elements - where n=time_thresh and k=the number of time steps we 
        # actually have        
        if ntime <(time_thresh-1):
            #print "Opening next file. DoY = %.0f" % np.min(grid_times[t_idx])
            df=Dataset(next_file,"r")
            hi_out[ii,ntime:]=\
            df[vname][:(time_thresh-ntime),rows[ii],cols[ii]]
            
        if np.max(hi_out[ii,:]) >=heat_thresh:
            hit_grid[rows[ii],cols[ii]]=1
            
    # Return all the hi_out values, and return the hit_grid 
    return hi_out, hit_grid



def conDOY_deg(time): 
    
    """
    
    Takes a netCDF4 time object and returns the doy in degrees or 
    radians. This is required to evaluate proximity between HWs and TCs - 
    independent of calendar.
    
    """
    
def filter_tracks(f,fo,yr_st,yr_stp): 
    
    """ 
    
    
    Iterate over the lines in the file and iterate over each year. 
    For each year, write an output file for the all the tracks.  
    
    """
    
    yr=np.nan
    
    with open(f,"r") as fi:
        
        lines=fi.readlines()
        
        for y in range(yr_st,yr_stp+1):
            
            with open(fo+"%.0f.txt"%y,"w") as fout:
                
                for line in lines:
                    
                    l=line.split()
                    
                    if len(l)==3:
                        try:
                            yr=np.float(l[0])
                        except:
                            pass
                        
                    if yr == y:
                        fout.write(line)
    
    
def parse_tracks_file(f,thresh):
    
    """
    
    Takes a TC file and checks all tracks for landfall and wind speed 
    exceedance. If both criteria are met, the lat/lon/date/wind are written out
    """
    count=0
    with open(f,"r") as fi:
        
        lines=fi.readlines()
        out=np.zeros((len(lines),5))
                        
        for line in lines:
                    
            l=line.split()
                    
            if len(l)<=3:
                try: i_d=np.int(l[1])
                except:
                    pass
                continue
                   
            wind_speed=np.float(l[4])
            land=np.int(l[-1])
                        
            if wind_speed>=thresh and land == 1:
                lon=np.float(l[2])
                if lon >180: lon-=360 # Correct to degrees East 
                out[count,0]=i_d # id
                out[count,1]=np.float(l[1]) # time
                out[count,2]=lon # lon(!)
                out[count,3]=np.float(l[3]) # lat
                out[count,4]=wind_speed # wind speed(!)
                            
                count+=1
    return out[:count,:]

def six2day(tc_decimal):
    
    """
    Convenience function to reduce a year's worth of six-hour TC data
    to occurrence/absence at daily resolution. The function rounds all 
    decimal dates to the nearest integer and returns the unique values
    of that
    """
    
    tc_day=np.unique(np.floor(tc_decimal))
    return tc_day

@jit  
def kernel_doy(date_decimal,tc_id,nyrs,stdv,nd):
    
    """
    
    Employs a normal kernel to estimate the probability of doy landfall. 
    See Brooks et al. (Tornado paper)
    
    - Calculate mean freq as f(doy)
    - Make periodic by reflecting
    - Apply Gaussian kernel 
    
    Note: Can be used for both h/w frequency and TC frequency 
    
    """
    
    # First coerce nyrs and stdv to floats
    nyrs=np.float(nyrs)
    stdv=np.float(stdv)
       
    # Double loop
    mu=np.zeros(nd)
    fn=np.zeros(nd)
    for n in range(1,nd+1):
        inner=np.zeros(nd)
        for k in range(nd+1):
            if n==1: mu[k-1]=len(np.unique(tc_id[date_decimal==k]))/nyrs # Compute this first run
            frac=np.abs((n-k))/np.float(nd)
            if frac >0.5: frac=1-frac
            nk=frac*nd
            inner[k-1]=mu[k-1]*np.exp(-0.5*np.power((nk/stdv),2))
        fn[n-1] = np.sum(inner)  
    fn/=(np.sqrt(2*np.pi)*stdv)
    
    return mu,fn
   
def kernel_smooth_doy(var,date_decimal,date_decimal_fit,stdv):
    
    """
    
    Employs a normal kernel to smooth day-of-year climatology 
    - Calculate mean as f(doy)
    - Apply Gaussian kernel 

    """
    
    # First coerce nyrs and stdv to floats
    stdv=np.float(stdv)
    doys=np.unique(date_decimal)
    nd=len(doys)   
    mu=np.zeros(nd)*np.nan
    fn=np.zeros(nd)*np.nan
    full=np.zeros(len(date_decimal_fit))*np.nan
    
    # Double loop    
    for n in range(nd):
        
        inner=np.zeros(nd)
        
        for k in range(nd):
            
            if n==0: mu[k]=np.nanmean(var[date_decimal==doys[k]]) # Compute this first run
            
            frac=np.abs((doys[n]-doys[k]))/np.float(nd)
            
            if frac >0.5: frac=1-frac
            
            nk=frac*nd
            
            inner[k]=mu[k]*np.exp(-0.5*np.power((nk/stdv),2))
            
        fn[n] = np.sum(inner)/(np.sqrt(2*np.pi)*stdv)
        full[date_decimal_fit==doys[n]]=fn[n]
        
    
    return mu,fn,full


def kernel_smooth_stat_doy(var,statin,date_decimal,date_decimal_fit,stdv):
    
    """
    
    Employs a normal kernel to smooth day-of-year climatology 
    for any stat in the numpy module. It does not deal with NaNs
    at present
    - Calculate stat as f(doy)
    - Apply Gaussian kernel 

    """

    # get the method
    stat=getattr(np,statin)
    
    # First coerce nyrs and stdv to floats
    stdv=np.float(stdv)
    doys=np.unique(date_decimal)
    nd=len(doys)   
    mu=np.zeros(nd)*np.nan
    fn=np.zeros(nd)*np.nan
    full=np.zeros(len(date_decimal_fit))*np.nan
    
    # Double loop    
    for n in range(nd):
        
        inner=np.zeros(nd)
        
        for k in range(nd):
            
            if n==0: mu[k]=stat(var[date_decimal==doys[k]]) # Compute this first run
            
            frac=np.abs((doys[n]-doys[k]))/np.float(nd)
            
            if frac >0.5: frac=1-frac
            
            nk=frac*nd
            
            inner[k]=mu[k]*np.exp(-0.5*np.power((nk/stdv),2))
            
        fn[n] = np.sum(inner)/(np.sqrt(2*np.pi)*stdv)
        full[date_decimal_fit==doys[n]]=fn[n]
        
    
    return mu,fn,full


def nc2text(ncfile,thresh,fo):
    
    """
    
    This function takes the IBTRACS TC file and writes to text file all those 
    landfalling TCs occurring in the Atlantic, with winds >thresh. 
    
    Slow - but won't be called often!
    
    """
    
    data=Dataset(ncfile,"r")
    nstorms = len(data.dimensions["storm"])
    time=data.variables["time"]
    ntime=time.shape[1]
    
    # Open file, iterate, and write out
    with open(fo,"w") as fo:
        
        for row in range(nstorms):
            
            # Temporarily hold all time steps of this TC and check for 
            # concurrency in TC+ status and landfall
            scratch_wind=data.variables["wmo_wind"][row].data[:]
            scratch_landfall=data.variables["landfall"][row].data[:]
            ind=np.logical_and(scratch_wind>=thresh,scratch_landfall==0)
            ind2=np.logical_and(ind,~np.isnan(scratch_wind))
            
            if ind2.any():

                ntime_i=np.sum(~data.variables["wmo_wind"][row].mask[:])
                
                for col in range(ntime_i):
                    
                    # Use this step to filter out spurious writing!
                    if data.variables["wmo_wind"][row,col].data>0:
                        # Deal with time
                         yr,mon,day,hr,dt=GF.conTimes(\
                         time_str=time.units,calendar="standard"\
                          ,times=np.atleast_1d(time[row,col].data),safe=False)
                         
                         # decimal time    
                         decday=dt2decday(np.atleast_1d(dt))[0]

                        # Write out
                         fo.write\
                         ("%.0f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.2f\t%.1f\n"%\
                         (row,yr,decday,data.variables["lon"][row,col].data,\
                         data.variables["lat"][row,col].data,\
                         data.variables["wmo_wind"][row,col].data,\
                         data.variables["wmo_pres"][row,col].data,\
                         data.variables["landfall"][row,col].data))     
                         
                     
                     
    return 0

def nc2text_pres(ncfile,thresh,fo):
    
    """
    
    This function takes the IBTRACS TC file and writes to text file all those 
    landfalling TCs occurring in the Atlantic, with press <=thresh. 
    
    Slow - but won't be called often!
    
    """
    
    data=Dataset(ncfile,"r")
    nstorms = len(data.dimensions["storm"])
    time=data.variables["time"]
    
    # Open file, iterate, and write out
    with open(fo,"w") as fo:
        
        for row in range(nstorms):
            
            # Temporarily hold all time steps of this TC and check for 
            # concurrency in TC+ status and landfall
            scratch_pres=data.variables["wmo_pres"][row].data[:]
            scratch_landfall=data.variables["landfall"][row].data[:]
            ind=np.logical_and(\
            np.logical_and(scratch_pres<=thresh,scratch_landfall==0),scratch_pres>0)
            ind2=np.logical_and(ind,~np.isnan(scratch_pres))
            
            if ind2.any():

                ntime_i=np.sum(~data.variables["wmo_pres"][row].mask[:])
                
                for col in range(ntime_i):
                    
                    # Use this step to filter out spurious writing!
                    if data.variables["wmo_pres"][row,col].data>0:
                        
                        # Deal with time
                         yr,mon,day,hr,dt=GF.conTimes(\
                         time_str=time.units,calendar="standard"\
                          ,times=np.atleast_1d(time[row,col]),safe=False)
                         
                         # Decimal time    
                         decday=dt2decday(np.atleast_1d(dt))[0]
                         
                         # Write out
                         str1="%.0f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.2f\t%.1f\t" \
                         % (row,yr[0],decday,data.variables["lon"][row,col],
                          data.variables["lat"][row,col],\
                          data.variables["wmo_wind"][row,col],\
                          data.variables["wmo_pres"][row,col],\
                          data.variables["landfall"][row,col])
 
                         
                         fo.write(str1+"\n") 
                     
    return 0

def nc2text_pres_basin(ncfile,thresh,fo):
    
    """
    
    This function takes the IBTRACS TC file and writes to text file all those 
    landfalling TCs occurring in the Atlantic, with winds >thresh. 
    
    Slow - but won't be called often!
    
    """
    
    data=Dataset(ncfile,"r")
    nstorms = len(data.dimensions["storm"])
    time=data.variables["time"]
    
    # Open file, iterate, and write out
    with open(fo,"w") as fo:
        
        for row in range(nstorms):
            
            # Temporarily hold all time steps of this TC and check for 
            # concurrency in TC+ status and landfall
            scratch_pres=data.variables["wmo_pres"][row].data[:]
            scratch_landfall=data.variables["landfall"][row].data[:]
            ind=np.logical_and(np.logical_and(scratch_pres<=thresh,\
                                          scratch_landfall==0),scratch_pres>0)
                             
            ind2=np.logical_and(ind,~np.isnan(scratch_pres))
            
            if ind2.any():

                ntime_i=np.sum(~data.variables["wmo_pres"][row].mask[:])
                
                for col in range(ntime_i):
                    
                    # Use this step to filter out spurious writing!
                    if data.variables["wmo_pres"][row,col].data>0:
                        
                        # Deal with time
                         yr,mon,day,hr,dt=GF.conTimes(\
                         time_str=time.units,calendar="standard"\
                          ,times=np.atleast_1d(time[row,col]),safe=False)
                         
                         # Decimal time    
                         decday=dt2decday(np.atleast_1d(dt))[0]
                         
                         # Write out
                         str1="%.0f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.2f\t%.1f\t" \
                         % (row,yr[0],decday,data.variables["lon"][row,col],
                          data.variables["lat"][row,col],\
                          data.variables["wmo_wind"][row,col],\
                          data.variables["wmo_pres"][row,col],\
                          data.variables["landfall"][row,col])
                         
                         str2=data.variables["basin"][row,col,0] + \
                          data.variables["basin"][row,col,1] + "\n" 
                         
                         fo.write(str1+str2) 
                         
                     
                     
    return 0            
#            for col in range(ntime):
#                
#                flag=False
#                    # Extract all time steps of this TC and check for 
#                    # concurrency in TC+ status and landfall
#                    scratch_wind=data.variables["wmo_wind"][:]
#                
#                if basin:
#                    
#                    # Extract all time steps of this TC and check for 
#                    # concurrency in TC+ status and landfall
#                    scratch_
#                    if data.variables["basin"][row,col]==basin and \
#                    data.variables["landfall"][row,col]==0 and \
#                    data.variables["wmo_wind"][row,col]>=thresh:
#                        flag = True
#                        
#                else:
#                    
#                    if data.variables["landfall"][row,col]==0 and \
#                    data.variables["wmo_wind"][row,col]>=thresh:
#                        flag = True
#                        
#                if flag:
#                     yr,mon,day,hr,dt=GF.conTimes(\
#                     time_str=time.units,calendar="standard"\
#                      ,times=np.atleast_1d(time[row,col]),safe=False)
#                     
#                     decday=dt2decday(np.atleast_1d(dt))[0]
#                    
#                     fo.write("%.0f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.2f\n"%\
#                     (row,yr,decday,data.variables["lon"][row,col],\
#                     data.variables["lat"][row,col],\
#                     data.variables["wmo_wind"][row,col],\
#                     data.variables["wmo_pres"][row,col]))         


 
def dt2decday(dt):
    
    
    """
    
    Converts python datetimes to decimal day of year
    
    """
    
    # Now compute the decimal day    
    dt=np.atleast_1d(dt); n=len(dt)
    try:
        dd=[dt[ii].timetuple().tm_yday+dt[ii].hour/24.0 for ii in range(n)]
    except:
        dd=[dt[ii].dayofyr+dt[ii].hour/24.0 for ii in range(n)]
       
    return dd




def calc_BC(obsNC,simNC,var,res_c,outNC):
    
    
    """
    Function takes two netcdf files of variable 'var' and: 
        
        - Calculates the percentiles for obs/sim (obs' and sim') for 0,(1-k),k 
        percentiles;where k = 1/res_c. Note: res_c=100 - 1percent increments; 
        res_c=1000 = 0.1percent intervals, and so forth...
        - Computes the bias (ac - for additive correction) in the percentiles 
        as obs'minus sim'
        - Writes sim' and ac to outNC
        
    """
    
    # Read in data
    sim_data=Dataset(simNC,"r")
    sim=sim_data[var]
    obs_data=Dataset(obsNC,"r")
    obs=obs_data[var]
    obs_lat=obs_data["lat"]
    obs_lon=obs_data["lon"]
    sim_lat=obs_data["lat"]
    sim_lon=obs_data["lon"]
    mask=obs[0,:,:].mask
    
    # Check for obvious errors
    assert sim.shape[1] == obs.shape[1] and sim.shape[2] == obs.shape[2],\
    "Input files have different dimension sizes! (along axes 1 and/or 2)"
    
    assert np.sum(obs_lat[:]-sim_lat[:])==0 and  np.sum(obs_lon[:]-sim_lon[:]) \
    ==0, "Coordinate axes differ!"
    
    # Set up percentile vector
    res=100/np.float(res_c)
    pcs=np.arange(1,100-res,res)    
    
    # Set missing value
    try:
        miss=sim._FillValue
    except:
        miss=obs._FillValue
            

    
    # Compute percentiles (axis=0), and the corrections
    count=0
    out_x=np.zeros((len(pcs),sim.shape[1],sim.shape[2]))
    out_ac=out_x*1.
    for pc in pcs:
        out_x[count,:,:]=np.percentile(sim,pc,axis=0)
        out_ac[count,:,:]=np.percentile(obs,pc,axis=0)-out_x[count,:,:]
        count+=1
    out_x[:,mask]=miss
    out_ac[:,mask]=miss
    
    # Prepare/write out
    varis={"x":[out_x,"degrees_Celsius"],"ac":[out_ac,"degrees_Celsius"]}
    # call function to write the nc file
    GF.write_nc_arb(outNC,varis,sim_lat,sim_lon,"percentile",\
    pcs,"percent",lat_string="",lon_string="",mv=miss)
    
    
    return 0  

def apply_BC(simNC,corrNC,var,cname,refname,outNC,retVal=False,prog=False):
    
    
    """
    
    Function takes a simulation file and a correction netCDF file 
    output from 'calc_BC' and uses these data to bias correct simNC
    This function wraps _apply_BC - which does the lifting
        
        -simNC   : netCDF file of simulation data to be corrected
        -corrNC  : netCDF file with the correction factors and their reference
        -cname   : string with name of the correction factors
        -refname : string with name of the reference values 
        
    """
       
    # Read in data
    sim_data=Dataset(simNC,"r")
    simgrid=sim_data[var]
    corr_data=Dataset(corrNC,"r")
    corrgrid=corr_data[cname][:,:,:]
    simref=corr_data[refname][:,:,:] 
    mask=corrgrid.mask
    missVal=corr_data[cname]._FillValue
    lat_out=sim_data["lat"]
    lon_out=sim_data["lon"]
    time_out=sim_data["time"]
       
    # Call function to correct
    out=_apply_BC(simgrid,corrgrid,simref,mask,missVal,prog)
    
    # Write out
    varis={"hi":[out,"degrees_Celsius"]}
    GF.write_nc(outNC,varis,lat_out,lon_out,time_out,\
             time_string="",lat_string="",\
             lon_string="",cal="",mv=missVal)
   
    if retVal: return out
    else: return 0

@jit
def _apply_BC(simgrid,corrgrid,simref,mask,missVal,prog=False):
    
    """
    
    This function implements an empirical bias correction. 
    
    It takes:
        -simgrid: model data to be corrected
        -corrgrid: an additive correction factor
        -simref: the reference values for which the corrections were defined
        -mask: True elements need not be evaluated; should speed up analysis
        -missVal: to be used for indicating missing data
        
    The mechanics: 
        -Iterate over the rows/cols of simgrid, and use values as
        xi interpolation points. The x and y interpolation points are
        the simref, and corrgrid variables, respectively. The yi interpol(and)
        is then added to simgrid. For simgrid>max(simref) or simgrid<min(simref)
        then the max/min yi are taken to add, as appropriate

    """
    
    # Meta info
    nt,nr,nc=simgrid.shape; ntot=np.float(nr*nc)
    out=np.ones((nt,nr,nc))*missVal
    
    # Quickly correct all simgrid values above and below max/mins
    ma=np.max(simref,axis=0)
    mi=np.min(simref,axis=0)

    # (Costly!) iteration
    counter=0
    for r in range(nr):
        
        for c in range(nc):
            
            # Skip if missing value (i.e. it's masked)
            if corrgrid.mask[0,r,c]:
                counter+=1
                continue

            # Index of those vals 'within' - simulation
            in_ind=np.logical_and(simgrid[:,r,c]>mi[r,c],\
                                  simgrid[:,r,c]<ma[r,c])
            in_ind_ref=np.logical_and(simref[:,r,c]>mi[r,c],
                                   simref[:,r,c]<ma[r,c])
            
            # Deal with correction factors for sims above/below
            ab_ind=simgrid[:,r,c]>=ma[r,c]
            bl_ind=simgrid[:,r,c]<=mi[r,c]
            
            if np.sum(ab_ind)>0:
                out[ab_ind,r,c]=simgrid[ab_ind,r,c]+corrgrid[-1,r,c] # last (largest)
            if np.sum(bl_ind)>0:    
               out[bl_ind,r,c]=simgrid[bl_ind,r,c]+corrgrid[0,r,c]  # first (smallest)
            
            # Correction factors for sims within range - interpolate 
            out[in_ind,r,c]=simgrid[in_ind,r,c]+np.interp(simgrid[in_ind,r,c],\
               simref[in_ind_ref,r,c],corrgrid[in_ind_ref,r,c])
            
            if prog: 
                counter+=1
                pct=counter/ntot*100.
                st="\r%.6f%% complete" % pct
                sys.stdout.write(st)
                sys.stdout.flush()

    return out


def get_gsod_data(master_dir,station_codes, startyr=1950, endyr=2018, parameters=None):

    """
    Parameters
    ----------
    station_codes : str or list
        Single station code or iterable of station codes to retrieve data for.
    start : ``None`` or date (see :ref:`dates-and-times`)
        If specified, data are limited to values after this date.
    end : ``None`` or date (see :ref:`dates-and-times`)

    Returns
    -------
    data_dict : dict
        Dict with station codes keyed to lists of value dicts.

    Credit: adpated from ulmo
    """

    data_dict = dict([(station_code, None) for station_code in station_codes])

    for year in range(startyr,endyr+1):
    	tar_path=master_dir+"/gsod_%.0f.tar"%year
    	with tarfile.open(tar_path, 'r:') as gsod_tar:
            stations_in_file = [
                	name.split('./')[-1].rsplit('-', 1)[0]
                	for name in gsod_tar.getnames() if len(name) > 1]
		# List of "hits" (stations we want; stations we can get)
            stations = list(set(station_codes) & set(stations_in_file))

	    for station in stations:
                year_data = _read_gsod_file(gsod_tar, station, year)
		if not year_data is None:
                    	if not data_dict[station] is None:
		
                       		data_dict[station] = np.append(data_dict[station], year_data)
                    	else:
                        	data_dict[station] = year_data

   
    for station, data_array in data_dict.items():
	
        if not data_dict[station] is None:
	    nt=len(data_dict[station])
            data_dict[station] = _record_array_to_value_dicts(data_array)
	    corevars = pd.DataFrame(data=np.array([[data_dict[station][ii]["dew_point"],\
                    		data_dict[station][ii]["mean_temp"],\
				data_dict[station][ii]["sea_level_pressure"]] \
				for ii in range(nt)]),\
				index=[data_dict[station][ii]["date"] for ii in range(nt)],\
				columns=["dew_point","mean_temp","slp"])
	else: corevars=np.array([0])

    return data_dict,corevars			



def _read_gsod_file(gsod_tar, station, year):
    tar_station_filename = station + '-' + str(year) + '.op.gz'
    try:
        gsod_tar.getmember('./' + tar_station_filename)
    except KeyError:
        return None
    
    ncdc_temp_dir = os.path.join('/tmp/', 'GSOD')
    if not os.path.isdir(ncdc_temp_dir): os.makedirs(ncdc_temp_dir)
    temp_path = os.path.join(ncdc_temp_dir, tar_station_filename)

    gsod_tar.extract('./' + tar_station_filename, ncdc_temp_dir)
    with gzip.open(temp_path, 'rb') as gunzip_f:
        columns = [
            # name, length, # of spaces separating previous column, dtype
            ('USAF', 6, 0, 'U6'),
            ('WBAN', 5, 1, 'U5'),
            ('date', 8, 2, object),
            ('mean_temp', 6, 2, float),
            ('mean_temp_count', 2, 1, int),
            ('dew_point', 6, 2, float),
            ('dew_point_count', 2, 1, int),
            ('sea_level_pressure', 6, 2, float),
            ('sea_level_pressure_count', 2, 1, int),
            ('station_pressure', 6, 2, float),
            ('station_pressure_count', 2, 1, int),
            ('visibility', 5, 2, float),
            ('visibility_count', 2, 1, int),
            ('mean_wind_speed', 5, 2, float),
            ('mean_wind_speed_count', 2, 1, int),
            ('max_wind_speed', 5, 2, float),
            ('max_gust', 5, 2, float),
            ('max_temp', 6, 2, float),
            ('max_temp_flag', 1, 0, 'U1'),
            ('min_temp', 6, 1, float),
            ('min_temp_flag', 1, 0, 'U1'),
            ('precip', 5, 1, float),
            ('precip_flag', 1, 0, 'U1'),
            ('snow_depth', 5, 1, float),
            ('FRSHTT', 6, 2, 'U6'),
        ]

        dtype = np.dtype([
            (column[0], column[3])
            for column in columns])

        # note: ignore initial 0
        delimiter = itertools.chain(*[column[1:3][::-1] for column in columns])
        usecols = list(range(1, len(columns) * 2, 2))

        data = np.genfromtxt(gunzip_f, skip_header=1, delimiter=delimiter,
                usecols=usecols, dtype=dtype, converters={5: _convert_date_string})
    os.remove(temp_path)

    # somehow we can end up with single-element arrays that are 0-dimensional??
    # (occurs on tyler's machine but is hard to reproduce)
    if data.ndim == 0:
        data = data.flatten()

    return data     

def _convert_date_string(date_string):
    if date_string == '':
        return None

    if isinstance(date_string, bytes):
        date_string = date_string.decode('utf-8')

    return datetime.datetime.strptime(date_string, '%Y%m%d').date()  

def _record_array_to_value_dicts(record_array):
    names = record_array.dtype.names
    value_dicts = [
        dict([(name, value[name_index])
                for name_index, name in enumerate(names)])
        for value in record_array]
    return value_dicts   
    
   

#=============================================================================#
# TESTING
#=============================================================================#
    
#yr_st=1986
#yr_stp=2015    
#di="/media/tom/WD12TB/TropicalCyclones/Consolidated/Stochastic/Tom Matthews/"
#f="/media/tom/WD12TB/TropicalCyclones/Consolidated/Stochastic/Tom Matthews/sim0000.dat"
#files=[ii for ii in os.listdir\
#("/media/tom/WD12TB/TropicalCyclones/Consolidated/Stochastic/Tom Matthews/") \
#if ".dat" in ii and ".gz" not in ii]
#do="/media/tom/WD12TB/TropicalCyclones/Consolidated/Stochastic/Tom Matthews/Filtered/"
#watch_file="/media/tom/WD12TB/TropicalCyclones/WATCH/merged_Atlantic_cal.nc"
#w=Dataset(watch_file,"r")
#grid_lon=w.variables["lon"]; grid_lat=w.variables["lat"]
##files=os.listdir(do)
#fin="/media/tom/WD12TB/TropicalCyclones/Allstorms.ibtracs_wmo.v03r10.nc"
##nc2text(fin,0,64,fo="OBS_ATLANTIC.txt")
#nctime=w.variables["time"]
#yr,mon,day,hr,dt=\
#    GF.conTimes(time_str=nctime.units,calendar=nctime.calendar,\
#                times=nctime[:],safe=False)
#    
## Select year
#idx=yr==1998
#hi=w.variables["hi"][idx,:,:] 
#dt_sel=dt[idx]
#grid_times=dt2decday(dt_sel)
#files=[ii for ii in os.listdir(do) ]
#test=np.concatenate(\
#       tuple([parse_tracks_file(do+files[ii],119.0) for ii in range(1)]))
#
#lons=test[:,2]; lats=test[:,3]; times=test[:,1]; 
#rows,cols=get_ll_rc(lons,lats,grid_lon,grid_lat)
#hi_out=get_hi(lons,lats,times,grid_lon,grid_lat,grid_times,30,hi)

#print "Read in 10k files..."
#doys=np.concatenate(\
#tuple([six2day(parse_tracks_file(do+files[ii],119.0)[:,1]) for ii in range(10000)]))
#mu_doy,kern_doy,test_k=kernel_doy(doys,10000,15,366)
#fig,ax=plt.subplots(1,1)
#ax.plot(mu_doy,color="blue")
#ax.plot(kern_doy,color="red")
##for count in range(len(files)):
#    fii=di+files[count]
#    fo=do+"file_%.0f_" % (count)
#    filter_tracks(fii,fo,yr_st,yr_stp)





