# Import the necessary modules
import argparse, sys
import pandas as 
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from osgeo import gdal, ogr
from rasterstats import zonal_stats
from scipy.stats import ks_2samp, ttest_rel
from sklearn.metrics import r2_score


def get_values(shapefile_filename, raster_filename, geometry_type):
    '''Extract the values from the shapefile and raster'''
    if geometry_type == "Polygon":
        # If polygon, do this:
        stats = zonal_stats(shapefile_filename, raster_filename, geojson_out=True)
        shapefile_value = pd.Series([x['properties'][shapefile_column] for x in stats])
        raster_value = pd.Series([x['properties']['mean'] for x in stats])
        return shapefile_value, raster_value
    else:
        # Otherwise, do this
        # Open the raster
        return ("I can't", "do this yet")
        raster_handle = gdal.Open(raster_filename) 
        geo_transform = raster_handle.GetGeoTransform()
        raster_band = raster_handle.GetRasterBand(1)

        # Open the shapefile
        shapefile_handle = ogr.Open(shapefile_filename)
        layer = shapefile_handle.GetLayer()

        # Extract the raster value for each feature in the shapefile
        for feature in layer:
            shapefile_value = feature.GetField(shapefile_column)
            geom = feature.GetGeometryRef()
            mx, my = geom.Centroid().GetX(), geom.Centroid().GetY()  #coord in map units
            #Convert from map to pixel coordinates.
            #Only works for geotransforms with no rotation.
            #If raster is rotated, see http://code.google.com/p/metageta/source/browse/trunk/metageta/geometry.py#493
            pixel_x = int((mx - geo_transform[0]) / geo_transform[1]) #x pixel
            pixel_y = int((my - geo_transform[3]) / geo_transform[5]) #y pixel
            raster_value = raster_band.ReadAsArray(pixel_x,pixel_y,1,1)
            print raster_value[0], shapefile_value

def ecdf(shapefile_value, raster_value, output_directory):
    '''Build and plot an Empirical Cumulative Distribution Function'''
    ecdf_shape = sm.distributions.ECDF(shapefile_value)
    x_shape = np.linspace(min(shapefile_value), max(shapefile_value))
    y_shape = ecdf_shape(x_shape)
    plt.plot(x_shape, y_shape, label="Plots")
    ecdf_raster = sm.distributions.ECDF(raster_value)
    x_raster = np.linspace(min(raster_value), max(raster_value))
    y_raster = ecdf_raster(x_raster)
    plt.plot(x_raster, y_raster, label="Raster")
    plt.legend(loc='lower right', shadow=True)
    pdffile = '{0}/ecdf_plot.pdf'.format(output_directory)
    print "Plotting ECDF. Output saved to {0}.".format(pdffile)
    plt.savefig(pdffile)

def ks(shapefile_value, raster_value):
    '''Compute the Kolmogorov-Smirnov (KS) statistic'''
    ks, pvalue = ks_2samp(raster_value, shapefile_value)
    print "Kolmogorov-Smirnov statistic is: {0}. P-value is {1}.".format(ks, pvalue)
    return ks, pvalue

def scatterplot_gmfr(shapefile_value, raster_value, output_directory):
    '''Compute the Geometric Mean Functional Relationship Regression and plot it with a scatterplot'''
    m = np.tan((np.arctan((len(shapefile_value) * np.sum(shapefile_value * raster_value) - np.sum(shapefile_value) * np.sum(raster_value)) / (len(shapefile_value) * np.sum(shapefile_value ** 2) - np.sum(shapefile_value) ** 2)) + np.arctan(1 / ( (len(shapefile_value) * np.sum(shapefile_value * raster_value) - np.sum(raster_value) * np.sum(shapefile_value)) / (len(shapefile_value) * np.sum(raster_value ** 2) - np.sum(raster_value) ** 2)))) / 2)
    b = np.mean(raster_value) - m * np.mean(shapefile_value)
    maxvalue = max(max(raster_value), max(shapefile_value)) + 10
    plt.scatter(shapefile_value, raster_value)
    axes = plt.gca()
    axes.set_xlim([0,maxvalue])
    axes.set_ylim([0,maxvalue])
    plt.plot([0, maxvalue], [m * 0 + b, m * maxvalue + b], color='r', label="GMFR")
    plt.plot([0, maxvalue], [0, maxvalue], color='k', linestyle='-', linewidth=1)
    plt.legend(loc='lower right', shadow=True)
    pdffile = '{0}/scatterplot_gmfr_plot.pdf'.format(output_directory)
    print 'Plotting scatterplot and GMFR Regression. Output saved to {0}.'.format(pdffile)
    plt.savefig(pdffile)
    return b, m

def agreement_coefficients(shapefile_value, raster_value, b, m):
    '''Compute Agreement Coefficient (AC), Systematic AC (ACsys), and Unsystematic AC (ACuns)'''
    ssd = np.sum((shapefile_value - raster_value) ** 2)
    spod = np.sum((abs(np.mean(shapefile_value) - np.mean(raster_value)) + abs(shapefile_value - np.mean(shapefile_value))) * (abs(np.mean(shapefile_value) - np.mean(raster_value)) + abs(raster_value - np.mean(raster_value))))
    ac = ssd / spod
    y_hat = m * shapefile_value + b
    x_hat = (raster_value - b) / m
    spd_u = np.sum((abs(shapefile_value - x_hat)) * (abs(raster_value - y_hat)))
    spd_s = ssd - spd_u
    ac_sys = 1 - (spd_s / spod)
    ac_uns = 1 - (spd_u / spod)
    print "Agreement Coefficient is {0}. Systematic Agreement Coefficient is {1}. Unsystematic Agreement Coefficient is {2}.".format(ac, ac_sys, ac_uns)
    return ac, ac_sys, ac_uns

def rmse(shapefile_value, raster_value):
    '''Calculate the Root Mean Square Error'''
    rmse = np.sqrt(np.mean((raster_value - shapefile_value) ** 2))
    print "RMSE is {0}".format(rmse)
    return rmse

def r2(shapefile_value, raster_value):
    '''Calculate the coefficient of determination (r2)'''
    r2 = r2_score(shapefile_value, raster_value)
    print "Coefficient of Determination is {0}".format(r2)
    return r2

def t_test(shapefile_value, raster_value):
    '''Compute the difference between the means (t-test)'''
    t, p = ttest_rel(shapefile_value, raster_value)
    print "The t-statistic is {0}. The pvalue is {1}.".format(t, p)
    return t, p

if __name__ == '__main__':
    # raster_filename = 'carbon_per_pixel_orig_rasters_model_73001_55001.tif'
    # shapefile_filename = 'biomass_plots_subset_carbon_sum_polygon.shp'
    # Get the arguments to the script.
    p = argparse.ArgumentParser(prog="continuous_variables.py", description="A program that assesses continuous geospatial datasets.")
    p.add_argument('-s', '--shapefile', dest='shapefile_filename', required=True, help='The full path to the shapefile containing the reference data.')
    p.add_argument('-r', '--raster', dest='raster_filename', required=True, help='The full path to the raster to be evaluated.')
    p.add_argument('-o', '--output_directory', dest='output_directory', required=True, help='The full path to the output directory where the plot output will be created.')
    p.add_argument('-g', '--geometry_type', dest='geometry_type', required=True, help='The geometry type of the input shapefile.', choices=['Point', 'Polygon'])
    p.add_argument('-c', '--column', dest='shapefile_column', required=True, help='The column of the shapefile containing the reference variable.')
    args = p.parse_args()

    # Read the data
    shapefile_value, raster_value = get_values(args.raster_filename, args.shapefile_filename, args.geometry_type)

