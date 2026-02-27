python download_gdelt.py
python unzip.py  # This assumes you have ~16 cpu cores, if not, change the --jobs flag on parallel to a reasonable number instead of 8
# At this point, all data has been converted to parquet. The next file, partition.py, filters down to the locations, countries, and event codes of interest. Edit that file to change our default lat/lon bounding box to what is desired for your study.
python partition.py

# At this point, an sqlite database containing the data in an easy to aggregate format is available. This can be loaded into a pytorch dataset via the class in gdelt_dataset_createor, or preprocessed even further to filter down to only specific events and countries
# Change the bounding box and countries/events to suit your study. By default this file filters the data down into a few countries on a small spatial region and aggregates the data into a csv for ease of use.
python gdelt_dataset_creator.py