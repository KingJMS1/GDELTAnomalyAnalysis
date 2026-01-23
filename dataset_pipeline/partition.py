import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import tqdm
import itertools as it

import numpy as np

import pyproj as pp
import adbc_driver_sqlite.dbapi as sqlite

wgs84 = pp.CRS.from_epsg(4326)
partitioning = ds.FilenamePartitioning(pa.schema([("year", pa.int16()), ("month", pa.int16())]))
partitioning2 = ds.HivePartitioning(pa.schema([("Actor1CountryCode", pa.string()), ("year", pa.int16()), ("month", pa.int16())]))
dataset = ds.dataset("gdelt_northern_hemisphere_20_min_lat", format="parquet", partitioning=partitioning, partition_base_dir="gdelt_northern_hemisphere_20_min_lat/")

res = pc.value_counts(dataset.to_table(columns=["EventRootCode"])["EventRootCode"])
res2 = pc.value_counts(dataset.to_table(columns=["Actor1CountryCode"])["Actor1CountryCode"])


ignoreEventCode = ["---", "X", ""]
badEventCodes = pa.array([x["values"] for x in res.to_pylist() if (x["counts"] < 10000) or (x["values"] in ignoreEventCode)])
badCountryCodes = pa.array([x["values"] for x in res2.to_pylist() if (x["counts"] < 10000) or (x["values"] == "")])

lonmin = -10
lonmax = 150
latmin = 20
latmax = 75

lonPartitions = list(np.lib.stride_tricks.sliding_window_view(np.arange(lonmin, lonmax + 1, 10), 2))
latPartitions = list(np.lib.stride_tricks.sliding_window_view(np.arange(latmin, latmax + 1, 5), 2))
years = list(range(2015, 2026))
months = list(range(1, 13))

allCountryCodes = [x["values"] for x in res2.to_pylist() if x["values"] not in badCountryCodes.to_pylist()]
allEventCodes = [x["values"] for x in res.to_pylist() if x["values"] not in badEventCodes.to_pylist()]


lat = ds.field("ActionGeo_Lat").cast(pa.float32())
lon = ds.field("ActionGeo_Lat").cast(pa.float32())

ignoreRows = pc.is_in(ds.field("EventCode"), badEventCodes) | pc.is_in(ds.field("Actor1CountryCode"), badCountryCodes) | (lat < latmin) | (lat > latmax) | (lon < lonmin) | (lon > lonmax)
less = dataset.filter(~ignoreRows)

length = len(years) * len(months) * len(lonPartitions) * len(latPartitions) * len(allCountryCodes) * len(allEventCodes)
lengthSmall = len(years) * len(months) * len(lonPartitions) * len(latPartitions)#* len(allCountryCodes)

pbar = tqdm.tqdm(total=lengthSmall)


def gen_data():
    with sqlite.connect("gdelt_europe2.db") as conn:
        with conn.cursor() as cur:
            for year in years:
                yearFilter = ds.field("year") == year
                y_filtered = less.filter(yearFilter)
                
                for month in months:
                    monthFilter = ds.field("month") == month
                    m_filtered = y_filtered.filter(monthFilter)

                    if year == 2015 and month < 2:
                        continue
                    
                    for lonPartition in lonPartitions:
                        lon = ds.field("ActionGeo_Long").cast(pa.float32())
                        lonFilter = (lonPartition[0] <=  lon) & (lon < lonPartition[1]) 
                        lonAvg = round(float(lonPartition[0] + lonPartition[1]) / 2)
                        lon_filtered = m_filtered.filter(lonFilter)

                        for latPartition in latPartitions:
                            lat = ds.field("ActionGeo_Lat").cast(pa.float32())
                            latFilter = (latPartition[0] <= lat) & (lat < latPartition[1])
                            latAvg = round(float(latPartition[0] + latPartition[1]) / 2)
                            
                            lat_filtered = lon_filtered.filter(latFilter)
                            fullBatch = lat_filtered.to_table()
                            if fullBatch.num_rows == 0:
                                pbar.update(1)
                                continue
                            fullBatch = fullBatch.append_column("latAvg", pa.array(it.repeat(latAvg, fullBatch.num_rows)))
                            fullBatch = fullBatch.append_column("lonAvg", pa.array(it.repeat(lonAvg, fullBatch.num_rows)))
                            cur.adbc_ingest("gdelt", fullBatch, "create_append")
                            pbar.update(1)

                    # for countryCode in allCountryCodes:
                    # countryFilter = ds.field("Actor1CountryCode") == countryCode
                    # country_filtered = m_filtered.filter(countryFilter)
                    # for batch in m_filtered.to_batches():
                    #     cur.adbc_ingest("gdelt", batch, "create_append")
                        # yield batch
                    # pbar.update(1)
        conn.commit()

gen_data()

# ds.write_dataset(gen_data(), "final", partitioning=partitioning2, schema=dataset.schema, format="parquet")

# pbar.close()
# chunker.close()