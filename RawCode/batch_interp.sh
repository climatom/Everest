#!/bin/bash
# This shell script will be called called via a LOTUS job array. 
# The command line will feature an argument (number) which tells the
# script which input file to read from. The input files themselves
# hold the names of netcdf paths. These will be passed to CDO for 
# horizontal interpolation and merging in time. Another program 
# will be used to check for data completeness and for vertical 
# interpolation/time averaging 

no=$1
no=$((no-1))
input="/group_workspaces/jasmin2/ncas_generic/users/tmatthews/EVEREST/input_${no}.txt"
odir="/group_workspaces/jasmin2/ncas_generic/users/tmatthews/EVEREST/HorInterp/"
count=0
while read fin; do

  inst="$(cut -d'/' -f7 <<<$fin)"
  model="$(cut -d'/' -f8 <<<$fin)"
  var="$(cut -d'/' -f15 <<<$fin)"
  oname="${odir}${inst}_${model}_${var}.nc"
  echo "${oname}" >${odir}/test${no}.txt
  
  cmd="cdo -L -s remapbil,lon=86.9250_lat=27.9881 $fin ${odir}input_${no}_scratch_${count}.nc"
  ${cmd}

  count=$((count+1))

done <$input


oname="${odir}${inst}_${model}_${var}.nc"
cmd="cdo -L -s mergetime ${odir}input_${no}_*scratch_*.nc ${oname}"
${cmd}
rm ${odir}input_${no}_*scratch_*.nc
