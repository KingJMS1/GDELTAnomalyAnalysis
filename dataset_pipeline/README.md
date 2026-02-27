### GDELT Dataset aggregation files
To use:

Step 1: Download raw gdelt2 files:
```bash
python download_gdelt.py
```

Step 2: 
Some files will probably fail to download the first time due to network issues/etc. Run the follwing file to retry failed downloads if desired.
```bash
python retry_failures.py
```

Step 3:
Unzip all the files, convert to monthly parquet files
```bash
python unzip.py
```

Step 4:
Select a reasonable spatial bounding box for your study by first editing the bounding box contained in the following file, and ensuring our minimmum event number per country and event codes are not too low for your study. Then run the file:
```bash
python partition.py
```

Step 5:
Everything is now in a sqlite database with easy querying and aggregation. We have written a pytorch dataset for this database in ```gdelt_dataset_creator.py```, ensure the spatial bounding box matches yours when using. To obtain a csv that can be used the rest of our modeling pipeline, run the freeze function to convert everything to a nice csv. 