# Continuous Variables
A python tool for evaluating the accuracy of maps of continuous variables.

## Dependencies
This script requires the following python libraries. These can all be installed
with "pip" or your favorite python package manager.

matplotlib.pyplot  
numpy  
osgeo  
pandas  
patsy
rasterstats  
scipy.stats  
sklearn.metrics (from sklearn)
statsmodels.api (from statsmodels)

On Linux/OSX, this should work to get the packages installed.
```bash
sudo pip install matplotlib numpy osgeo pandas patsy rasterstats scipy sklearn statsmodels
```

## Version 2.0
This is a standalone python script that implements some of the measures described
in R. Riemann et al. / Remote Sensing of Environment 114 (2010) 2337–2352. This is 
a generic implementation and this script can be further customized by the end user
to include graph titles, labels, legends, etc that are specific to their dataset.
This script DOES NOT provide any GIS preprocessing. It is expected that the user 
will have prepared an appropriate raster file and shapefile that properly overlay 
and have the same projection. If your plot data are represented by points, you 
should consider buffering those points to represent your actual plot area and using
the polygon option for a more appropriate comparison. It is assumed that the user
understands the Briemann et al. methodology. This script presents the results of 
the calculations and provides no interpretation of those results.

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
                        The name of the column in the shapefile containing the reference
                        variable.
```

## Changelog
* Added argparse module to handle help argument passing
* Added support for point shapefiles
* PDF output from plotting functions
* More detailed code comments