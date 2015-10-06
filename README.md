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

## Version 2.0
This is a standalone python script that implements some of the measures described
in R. Riemann et al. / Remote Sensing of Environment 114 (2010) 2337–2352.

```
usage: continuous_variables.py [-h] -s SHAPEFILE_FILENAME -r RASTER_FILENAME
                               -o OUTPUT_DIRECTORY -g {Point,Polygon} -c
                               SHAPEFILE_COLUMN  

A program that assesses continuous geospatial datasets.  

optional arguments:
  -h, --help            show this help message and exit
  -s SHAPEFILE_FILENAME, --shapefile SHAPEFILE_FILENAME
                        The full path to the shapefile containing the
                        reference data.
  -r RASTER_FILENAME, --raster RASTER_FILENAME
                        The full path to the raster to be evaluated.
  -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        The full path to the output directory where the plot
                        output will be created.
  -g {Point,Polygon}, --geometry_type {Point,Polygon}
                        The geometry type of the input shapefile.
  -c SHAPEFILE_COLUMN, --column SHAPEFILE_COLUMN
                        The column of the shapefile containing the reference
                        variable.
```

