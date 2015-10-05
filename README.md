# Continuous Variables
A python tool for evaluating the accuracy of maps of continuous variables.

## Dependencies
This script requires the following python libraries. These can all be installed
with "pip" or your favorite python package manager.

matplotlib.pyplot

numpy

osgeo

pandas

rasterstats

scipy.stats

sklearn.metrics

statsmodels.api


## Version 1.0
This is a standalone python script that implements some of the measures described
in R. Riemann et al. / Remote Sensing of Environment 114 (2010) 2337–2352.

The script takes 4 arguments. The first is a raster file of a mapped continuous 
variable (biomass, etc). The second is a shapefile that contains the reference 
data. For now, this must be a polygon that represents the plot area. The third is
the type of geometry. For now, this MUST BE 'Polygon'. The fourth is the name of
the attribute in the shapefile that matches the raster data set. 


