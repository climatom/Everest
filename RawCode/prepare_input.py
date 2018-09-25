#!/usr/bin/python2.7
#####################################################################
# This script takes an arbitary number of variables and experiments 
# and checks the BADC archives for model runs where all variables and
# all experiments are available (i.e. held in the archive). The first 
# part of this script is totally transferrable; the second is an 
# example of how this list can be used (in this case, the files are 
# iterated over and each (variable/experiment) is interpolated to the 
# location of Mount Everest. All files for each variable/experiment 
# are then merged!
#####################################################################


#####################################################################
#====================================================================
# Part 1
#====================================================================
#####################################################################

# Import modules
import os,sys,numpy as np

# Template path: 
#/badc/cmip5/data/cmip5/output1/BCC/bcc-csm1-1/historical/day/atmos/day/r1i1p1/latest/zg/
base="/badc/cmip5/data/cmip5/output1/"
mid="/day/atmos/day/r1i1p1/latest/"
vars=["zg","ta"]
exps=["historical","rcp85"]

# Find all modelling groups that satisfy all the paths...	
groups=os.listdir(base)
tree=[]; n=0; confirmed_mods=[]; ni=0; 
for g in groups: 
	mods=os.listdir(base+g)	
	for m in mods:
		path=base+g+"/"+m+"/"+exps[0]+mid+vars[0]
		scratch_array=[]; scratch_name=""
		if os.path.isdir(path): # then historical var 1 exists. 
			# iterate over the remaining combinations to check for full availability
			ni=1
			for v in vars:
				for e in exps:
					if v==vars[0] and e==exps[0]: continue
					scratch_name=path.replace(exps[0],e).replace(vars[0],v)
					if os.path.isdir(scratch_name): ni+=1; scratch_array.append(scratch_name); 
			if ni == len(vars)+len(exps): n+=1; tree+=list(set(scratch_array)); 
						
		
print "Checking complete: %02d model runs fulfil criteria" % n; print len(tree)

#####################################################################
#====================================================================
# Part 2
#====================================================================
#####################################################################

# Write out the full file list for each model

# Output directory  
odir="/group_workspaces/jasmin2/ncas_generic/users/tmatthews/EVEREST/"

# Begin by looping over the full path list 
count=0
for p in tree:
	if "historical" in p: print p; continue # Do this because we'll use the rcp
	# file names to identify the correspoding historical model runs - 
	# which we already know exist. 
	# Open text file to write file list to
	with open(odir+"input_%.0f.txt"%count,"w") as fo:	

		# List all files in the (rcp) folder
		files=[p+"/"+ii for ii in os.listdir(p) if ".nc4" not in ii]
		
		# Find the historical path -- whatever the rcp
		for r in ["rcp26","rcp45","rcp60","rcp85"]: hist_p=p.replace(r,"historical")
		files+=[hist_p+"/"+ii for ii in os.listdir(hist_p)] 
		
		# Now iterate over all these rcp/model files...
		nfi=0
		for f in files: fo.write(f+"\n")
	
	
		count+=1; 



