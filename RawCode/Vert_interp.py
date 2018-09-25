import numpy as np
import os
from netCDF4 import Dataset

import numpy as np, pandas as pd
import os, netCDF4
from netCDF4 import Dataset

# Function to write netcdf object(s) --> netcdf objects 
def write_nc(ofile,values,varnames,varunits,time_out):

  # Create file  
  ncfile=Dataset(ofile,"w")
  
  # Create time dimension
  ntime=len(time_out)
  ncfile.createDimension('time',ntime)

  # Define the coordinate variables. They will hold the coordinate
  # information 
  times = ncfile.createVariable('time',np.float32,dimensions=['time',])
  times.units = time_out.units
  times.calendar=time_out.calendar     
  # write data to coordinate vars.
  times[:] = time_out[:]
 
  # now create variable
  for ii in range(len(varnames)):
  	var = ncfile.createVariable(varnames[ii],np.float32,dimensions=['time',])
  	var[:] = values[ii]
  	var.units = varunits[ii]
     
  # now close file 
  ncfile.close() 


# Function to check the integrity of a CMIP5 model - 
# Must have no more than max_dt years between entries 
def check_data(infile,year_min,year_max,max_dt):

	fobj=Dataset(infile,"r")
	time=fobj.variables["time"]
	pdate=netCDF4.num2date(time[:],units=time.units,calendar=time.calendar)
	year=[ii.year for ii in pdate]
	dt=np.diff(year)
	if np.min(year)<=year_min and np.max(year)>=year_max and np.max(dt) <= max_dt:

		return 0 
	else:
		return 999



# In directory 
di="/group_workspaces/jasmin2/ncas_generic/users/tmatthews/EVEREST/HorInterp/"

# Out directory
do="/group_workspaces/jasmin2/ncas_generic/users/tmatthews/EVEREST/VerInterp/"

# List all zg files
fis=[ii for ii in os.listdir(di) if ".nc" in ii and "zg" in ii]

# Loop over and interpolate to Everest 
ht=8848.0 
for r in fis:
	df=Dataset(di+r,"r")
	plev=df.variables["plev"][:]
        assert plev[0]>plev[-1],"Plev not ascending!"
	time=df.variables["time"]; nt=len(time)
	z=np.squeeze(df.variables["zg"][:,:])
	
	if check_data(di+r,1981,2099,1) != 0: continue

	try:
		pE=np.array([np.interp(np.atleast_1d(ht),\
		z[ii,~z[ii,:].mask],plev[~z[ii,:].mask]) for ii in range(nt)])
	except:
		pE=np.array([np.interp(np.atleast_1d(ht),\
		z[ii,:],plev[:]) for ii in range(nt)])

	print "Interpolated pressure in file: %s" % r

	# --------------------------------------------------#
	# Repeat 
	# --------------------------------------------------#

	df2=Dataset(di+r.replace("zg","ta"),"r")
	ta=np.squeeze(df2.variables["ta"][:,:])
        assert plev[0]>plev[-1],"Plev not ascending!"
	time=df2.variables["time"]; nt=len(time)
	assert z.shape == ta.shape

	
	if check_data(di+r,1981,2099,1) != 0: continue	
	tE=np.zeros(nt)*np.nan
	for ii in range(nt):
		z_scratch=np.squeeze(z[ii,:])
		ta_scratch=np.squeeze(ta[ii,:])
		# Now order z and ta
		order=np.argsort(z_scratch)
		z_order=z_scratch[order]
		ta_order=ta_scratch[order]
		try:
			tE[ii]=np.interp(np.atleast_1d(ht),\
			z_order[~z_order[ii,:].mask],ta_order[~z_order[ii,:].mask]) 
		except:
			tE[ii]=np.interp(np.atleast_1d(ht),\
			z_order,ta_order) 			


	# Write out
	oname=do+r.replace("zg","").replace(".nc","rcp85.nc")
	write_nc(ofile=oname,values=[pE,tE],varnames=["Pressure","Temp"],varunits=["Pa","Kelvin"],time_out=time)

	print "Interpolated temp and finished with file %s" % r



