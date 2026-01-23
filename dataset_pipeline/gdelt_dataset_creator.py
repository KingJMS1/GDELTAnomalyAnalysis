import itertools as it
import datetime as dt
import sqlite3

import torch as pt
import pandas as pd
import numpy as np

import pandas as pd
import torch as pt
import itertools as it
import numpy as np
import tqdm

# This class can also be used standalone if you do not want to freeze to a csv
class GDELTCountDataCreator(pt.utils.data.Dataset):
    def __init__(self, countries = None, eventData = None, lookback = 10, lonmin=-5, lonmax = 145, latmin = 22, latmax = 72, startyear = 2015, endyear = 2025):
        super(GDELTCountDataCreator).__init__()
        
        # Setup database connection
        self.conn = sqlite3.connect("gdelt_europe2.db")
        self.cur = self.conn.cursor()
        
        # If we are not initialized with  or events, initialize from defaults
        self.default_countries, self.default_eventData, self.londata, self.latdata = GDELTCountDataCreator.defaults()
        self.londata = [x for x in self.londata if (x >= lonmin) and (x <= lonmax)]
        self.latdata = [x for x in self.latdata if (x >= latmin) and (x <= latmax)]
        self.countries = countries
        self.eventcol = None
        self.events = None

        # Create spatial partitions
        self.spatial_partitions = list(it.product(self.londata, self.latdata))

        # Default event data
        if eventData is None:
            self.eventcol, self.events = self.default_eventData
        else:
            self.eventcol, self.events = eventData
        
        # Default country data
        if self.countries is None:
            self.countries = self.default_countries

        # Setup spatiotemporal bounds
        self.minyear = startyear
        self.maxyear = endyear


        # Create temporal limits
        self.timeformat = "%Y%m%d%H%M%S"
        self.startdate = None
        self.enddate = dt.date(self.maxyear + 1, 1, 1)
        if self.minyear == 2015:
            self.startdate = dt.date(self.minyear, 2, 18)
        
        # Create temporal partitions
        self.weekly_range = pd.date_range(start=self.startdate, end=self.enddate, freq='W')
        self.time_partitions = np.lib.stride_tricks.sliding_window_view(self.weekly_range.to_numpy(), lookback + 2)
        self.lookback = lookback

        self.colnames = []

        for country in self.countries:
            for event in self.events:
                for space_partition in self.spatial_partitions:
                    lon, lat = space_partition
                    self.colnames.append(f"{country}_{event}_{lon}_{lat}")

        self.sql = f"""
        select COUNT(ID), Actor1CountryCode, {self.eventcol} from gdelt
        where
            (year = ? or year = ?) and
            (month = ? or month = ?) and
            
            (lonAvg = ?) and
            (latAvg = ?) and
            Actor1CountryCode in {tuple(self.countries)} and
            {self.eventcol} in {tuple(self.events)} and

            (DATEADDED >= ? AND DATEADDED <= ?) 

        group by Actor1CountryCode, {self.eventcol}
        order by Actor1CountryCode, {self.eventcol}
        ;
        """


    @staticmethod
    def defaults():
        countries = ['CHN', 'CUB', 'USA', 'SAU', 'MEX', 'FRA', 'MDV', 'THA', 'TWN', 'IND', 'MAR', 'KOR', 'VNM', 'SAS', 'LAO', 'AFR', 'URY', 'VEN', 'CHE', 'JPN', 'OMN', 'SEN', 'YEM', 'BGD', 'GBR', 'IRN', 'ISR', 'NPL', 'PAK', 'AFG', 'ESP', 'PAN', 'CAN', 'NMR', 'SYC', 'BLZ', 'GTM', 'HND', 'SLV', 'PRT', 'ARE', 'MMR', 'FIN', 'SYR', 'PHL', 'ITA', 'GRC', 'AUS', 'KEN', 'RUS', 'BHR', 'DEU', 'EGY', 'EUR', 'IRQ', 'JOR', 'KWT', 'LBN', 'LBY', 'LKA', 'MYS', 'NZL', 'QAT', 'SGP', 'SOM', 'TUR', 'TZA', 'ARG', 'VAT', 'NOR', 'GHA', 'KHM', 'NLD', 'POL', 'SEA', 'ZMB', 'BHS', 'NAM', 'AUT', 'BEL', 'BRA', 'CMR', 'CZE', 'DNK', 'ECU', 'FJI', 'HKG', 'IDN', 'IRL', 'LUX', 'MCO', 'MHL', 'NGA', 'PLW', 'PSE', 'SDN', 'SWE', 'TON', 'UKR', 'WSM', 'ZAF', 'ALB', 'PER', 'PRK', 'HUN', 'PGS', 'ATG', 'AZE', 'CHL', 'COL', 'CRB', 'CYM', 'DOM', 'DZA', 'ERI', 'GIN', 'HTI', 'JAM', 'LBR', 'LCA', 'SLE', 'SVK', 'TCD', 'TUN', 'UGA', 'UZB', 'DJI', 'MNG', 'NER', 'NIC', 'BRN', 'GMB', 'MAC', 'GRD', 'BMU', 'LTU', 'MOZ', 'PNG', 'GUY', 'WST', 'BTN', 'TTO', 'MLI', 'COD', 'BGR', 'CAF', 'CRI', 'ETH', 'ISL', 'VCT', 'BOL', 'BRB', 'DMA', 'ZWE', 'BWA', 'ASA', 'BLR', 'EST', 'KNA', 'KAZ', 'BEN', 'SRB', 'KGZ', 'ARM', 'AGO', 'MEA', 'TJK', 'CYP', 'LVA', 'MLT', 'CIV', 'MWI', 'SSD', 'MRT', 'COG', 'GNQ', 'HRV', 'MDG', 'RWA', 'MKD', 'VUT', 'MUS', 'TMP', 'SLB', 'TKM', 'CPV', 'GAB', 'WAF', 'BFA', 'CAS', 'SUR', 'MDA', 'TGO', 'BDI', 'PRY', 'LSO', 'LIE', 'GEO', 'SMR', 'AND',]
        eventData = ("EventRootCode", [f"{x:02}" for x in range(1, 21)])
        lons = list(range(-5, 146, 10))
        lats = list(range(22, 73, 5))

        return countries, eventData, lons, lats


    def __len__(self):
        return len(self.time_partitions)


    def __getitem__(self, index):
        # Each row is a target week
        time_partition = self.time_partitions[index]

        weeks_dt = pd.Series(time_partition)
        weeks_int = weeks_dt.dt.strftime(self.timeformat).astype(int)
        years = weeks_dt.dt.year
        months = weeks_dt.dt.month

        yearlims = np.lib.stride_tricks.sliding_window_view(years, 2)
        monthlims = np.lib.stride_tricks.sliding_window_view(months, 2)
        weeklims = np.lib.stride_tricks.sliding_window_view(weeks_int, 2)
        
        temporal_partitions = list(zip(yearlims, monthlims, weeklims))
        
        series = {x: np.zeros(self.lookback + 1) for x in self.colnames}

        for i, temp_partition in enumerate(temporal_partitions):
            year, month, week = temp_partition
            yearmin, yearmax = year[0].item(), year[1].item()
            monthmin, monthmax = month[0].item(), month[1].item()
            weekmin, weekmax = week[0].item(), week[1].item()
            
            for space_partition in self.spatial_partitions:
                lon, lat = space_partition
                self.cur.execute(self.sql, (yearmin, yearmax, monthmin, monthmax, lon, lat, weekmin, weekmax))
                data = self.cur.fetchall()
                for count, country, event in data:
                    series[f"{country}_{event}_{lon}_{lat}"][i] = count
        
        allData = pd.DataFrame(series).to_numpy()
        X = pt.tensor(allData[:-1], dtype=pt.float32)
        y = pt.tensor(allData[-1], dtype=pt.float32)
        
        return X, y
        
    def freeze(self, outfile = "gdelt.csv"):
        # Each row is a target week
        time_partition = self.weekly_range.to_numpy()

        weeks_dt = pd.Series(time_partition)
        weeks_int = weeks_dt.dt.strftime(self.timeformat).astype(int)
        years = weeks_dt.dt.year
        months = weeks_dt.dt.month

        yearlims = np.lib.stride_tricks.sliding_window_view(years, 2)
        monthlims = np.lib.stride_tricks.sliding_window_view(months, 2)
        weeklims = np.lib.stride_tricks.sliding_window_view(weeks_int, 2)
        
        temporal_partitions = list(zip(yearlims, monthlims, weeklims))
        series = {x: np.zeros(len(temporal_partitions)) for x in self.colnames}

        pbar = tqdm.tqdm(total=len(temporal_partitions) * len(self.spatial_partitions))

        for i, temp_partition in enumerate(temporal_partitions):
            year, month, week = temp_partition
            yearmin, yearmax = year[0].item(), year[1].item()
            monthmin, monthmax = month[0].item(), month[1].item()
            weekmin, weekmax = week[0].item(), week[1].item()
            
            for space_partition in self.spatial_partitions:
                lon, lat = space_partition
                self.cur.execute(self.sql, (yearmin, yearmax, monthmin, monthmax, lon, lat, weekmin, weekmax))
                data = self.cur.fetchall()
                for count, country, event in data:
                    series[f"{country}_{event}_{lon}_{lat}"][i] = count
                pbar.update(1)
        pbar.close()
        
        print("Writing to csv")
        pd.DataFrame(series, index=weeks_dt[:-1]).to_csv(outfile)
        

if __name__ == "__main__":
    data = GDELTCountDataCreator(
        ["USA", "RUS", "UKR", "ISR", "TUR", "PSE", "DEU"], 
        # ("EventCode", ["1211", "1212", "0212", "0232", "0234", "0256", "0356", "072", "073", "1012", "1123", "1222", "1224", "1382", "1384", "154", "192", "194"]), 
        ("EventRootCode", ["02", "03", "04", "06", "07", "09", "10", "11", "12", "13", "15", "16", "17", "18", "19"]),
        latmax=60, lonmin=15, lonmax=55
        )
    
    data.freeze()