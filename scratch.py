import pandas as pd
import shapely
import geopandas as gpd
import fiona
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

d2 = pd.read_csv("/Users/ibrugere/Downloads/CPS_School_Locations_SY1415.csv")

d1 = pd.read_excel("/Users/ibrugere/Downloads/Accountability_SQRPratings_2018-2019_SchoolLevel.xls", sheet_name=2, header=1)
d1 = d1.drop(0)
d=d1
de = pd.read_excel("/Users/ibrugere/Downloads/Accountability_SQRPratings_2018-2019_SchoolLevel.xls", sheet_name=1, header=1)
de = de.drop(0)

d = pd.concat([d1, de])

d = d[(d["Network "].str.startswith("Network")) & (d["SY 2018-2019 SQRP Rating"].isin(["Level 1+"]))]

df = pd.merge(d, d2, left_on=  ['School ID'], right_on=["SCHOOL_ID"] ,how = 'inner')


ddf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.X, df.Y))

ddf.to_file("/Users/ibrugere/new_schools.kml", driver='KML')

ddf = ddf.rename(columns={"X": "long", "Y":"lat"})
ddf.to_csv("/Users/ibrugere/Desktop/graph_reinforcement/data/schools/schools_cps_highschool.csv")
