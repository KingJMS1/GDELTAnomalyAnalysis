import os
from traceback import print_exc
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import dataset, csv, fs

import tqdm

# Schema for csv files
colTypes = {
    "ID": pa.int64(),
    "Day": pa.timestamp("s"),
    "MonthYear": pa.int32(),
    "Year": pa.int32(),
    "FractionDate": pa.string(),
    "Actor1Code": pa.string(),
    "Actor1Name": pa.string(),
    "Actor1CountryCode": pa.string(),
    "Actor1KnownGroupCode": pa.string(),
    "Actor1EthnicCode": pa.string(),
    "Actor1Religion1Code": pa.string(),
    "Actor1Religion2Code": pa.string(),
    "Actor1Type1Code": pa.string(),
    "Actor1Type2Code": pa.string(),
    "Actor1Type3Code": pa.string(),
    "Actor2Code": pa.string(),
    "Actor2Name": pa.string(),
    "Actor2CountryCode": pa.string(),
    "Actor2KnownGroupCode": pa.string(),
    "Actor2EthnicCode": pa.string(),
    "Actor2Religion1Code": pa.string(),
    "Actor2Religion2Code": pa.string(),
    "Actor2Type1Code": pa.string(),
    "Actor2Type2Code": pa.string(),
    "Actor2Type3Code": pa.string(),
    "IsRootEvent": pa.int32(),
    "EventCode": pa.string(),
    "EventBaseCode": pa.string(),
    "EventRootCode": pa.string(),
    "QuadClass": pa.int32(),
    "GoldsteinScale": pa.float32(),
    "NumMentions": pa.int64(),
    "NumSources": pa.int64(),
    "NumArticles": pa.int64(),
    "AvgTone": pa.float32(),
    "Actor1Geo_Type": pa.int32(),
    "Actor1Geo_Fullname": pa.string(),
    "Actor1Geo_CountryCode": pa.string(),
    "Actor1Geo_ADM1Code": pa.string(),
    "Actor1Geo_ADM2Code": pa.string(),
    "Actor1Geo_Lat": pa.string(),
    "Actor1Geo_Long": pa.string(),
    "Actor1Geo_FeatureID": pa.string(),
    "Actor2Geo_Type": pa.int32(),
    "Actor2Geo_Fullname": pa.string(),
    "Actor2Geo_CountryCode": pa.string(),
    "Actor2Geo_ADM1Code": pa.string(),
    "Actor2Geo_ADM2Code": pa.string(),
    "Actor2Geo_Lat": pa.string(),
    "Actor2Geo_Long": pa.string(),
    "Actor2Geo_FeatureID": pa.string(),
    "ActionGeo_Type": pa.int32(),
    "ActionGeo_Fullname": pa.string(),
    "ActionGeo_CountryCode": pa.string(),
    "ActionGeo_ADM1Code": pa.string(),
    "ActionGeo_ADM2Code": pa.string(),
    "ActionGeo_Lat": pa.string(),
    "ActionGeo_Long": pa.string(),
    "ActionGeo_FeatureID": pa.string(),
    "DATEADDED": pa.int64(),
    "SOURCEURL": pa.string()
}
schema = pa.schema(colTypes)
colNames = list(colTypes.keys())
sep = "\t"

# Setup pyarrow options
ParseOptions = csv.ParseOptions(delimiter = sep)
ConvertOptions = csv.ConvertOptions(column_types = schema, timestamp_parsers = ["%Y%m%d"])
ReadOptions = csv.ReadOptions(column_names = colNames)
FileFormat = dataset.CsvFileFormat(parse_options = ParseOptions, convert_options = ConvertOptions, read_options = ReadOptions)

# Function to unzip and process all files within 1 year.
def unzip(folder):
    destFolder = folder + "_unzip"
    os.makedirs(destFolder, exist_ok = True)
    
    # Setup parquet writer
    writer = pq.ParquetWriter(f'{destFolder}/{folder}.parquet', schema)

    years = [str(x) for x in range(2015, 2027)]
    months = [f"{x:02}" for x in range(1, 13)]
    
    command = f"parallel --jobs 8 unzip {{}} -d {destFolder} ::: "

    for year in tqdm.tqdm(years):
        for month in months:
            if not any([x.startswith(year + month) for x in os.listdir(folder)]):
                continue
            try:
                currYearFiles = folder + "/" + year + month + "*.zip"
                subprocess.run(command + currYearFiles, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check = True, shell=True)
                currYearOutput = [destFolder + "/" + x for x in os.listdir(destFolder) if x.startswith(year)]
                currYearOutput = [x for x in currYearOutput if os.path.getsize(x) > 0]
                allData = dataset.FileSystemDataset.from_paths(currYearOutput, schema=schema, format=FileFormat, filesystem = fs.LocalFileSystem())
                toOut = allData.to_table()
                writer.write_table(toOut)
            except:
                print(f"Error processing year {year} month {month}. Exception: ")
                print_exc()
            finally:
                # Cleanup any leftover csvs
                subprocess.run(f"rm {destFolder}/*.CSV", shell=True)
    
    writer.close()

if __name__ == "__main__":
    folder = "exports"
    unzip(folder)