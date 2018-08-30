#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import datetime, os, calendar
from dateutil.relativedelta import relativedelta

server = ECMWFDataServer()

# Ask for 1 month data
st=datetime.datetime(year=1979,month=1,day=1)


while st <= datetime.datetime(year=2016,month=12,day=1):
	
	yr=st.year
	mon=st.month
	oname="/media/tom/WD12TB/Everest/ERA-I_%.02d_%.02d.nc"%(mon,yr)

	if os.path.isfile(oname):
		print "Already have file: %s; skipping..." %oname
		st=st+relativedelta(months=1)
		continue
	
	while True:

			print "\n\n\nTrying to download month %.02d, year %.02d..."%(mon,yr)
		        print "Will store in file: %s\n\n\n" % oname

			try:
				ndays_month=calendar.monthrange(st.year, st.month)[1]
				start_string="%02d-%02d-%02d" % (st.year,st.month,st.day)
				stop_string="%02d-%02d-%02d" % (st.year,st.month,ndays_month)
				date_string=start_string+"/to/"+stop_string
				print "*******************************************************************************"
				print "\t\t\tRetrieving: %s" % (date_string)
				print "*******************************************************************************"

				server.retrieve({

					"class": "ei",
					"dataset": "interim",
					"expver": "1",
					"stream": "oper",
					"type": "an",
					"levtype": "pl",
					"param": "129.128/130.128/131.128/132.128/133.128/135.128",
				    	"date": date_string,
					"time": "00:00:00/06:00:00/12:00:00/18:00:00",
				    	"levelist": "200/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
					"grid":"0.75/0.75",
					"area": "35/80/20/95.0",
					"format":"netcdf",
				    	"target": "scratch.nc",

				})
			
				print "\n\n\nGot data!\n\n\n"
				
				# Command to convert geopot to m
				cmd="cdo -L -s -O -b 32 -f nc divc,9.80665 -selvar,z scratch.nc scratch_geopot.nc"
				fail=os.system(cmd); assert fail==0,"This failed:\n%s"%cmd
				cmd="cdo -L -s -O -b 32 -f nc merge scratch.nc scratch_geopot.nc %s"%oname
				fail=os.system(cmd); assert fail==0,"This failed:\n%s"%cmd
				
			except:
				continue
			break	

	st=st+relativedelta(months=1)



