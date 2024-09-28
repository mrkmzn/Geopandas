import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# crime data from ISTAT - data from 2021 (http://dati.istat.it/Index.aspx?QueryId=25097&lang=en)
crime1 = pd.read_csv('crime1.csv')

# remove regions from dataframe to plot only municipalities (I mistakenly downloaded aggregated data of regions)
regions = [
    "Abruzzo", 
    "Basilicata", 
    "Calabria", 
    "Campania", 
    "Emilia-Romagna", 
    "Friuli Venezia Giulia", 
    "Lazio", 
    "Liguria", 
    "Lombardia", 
    "Marche", 
    "Molise", 
    "Piemonte", 
    "Puglia", 
    "Sardegna", 
    "Sicilia", 
    "Toscana", 
    "Trentino Alto Adige / Sudtirol", # I changed the name on the df not to deal with special characters
    "Umbria", 
    "Valle d'Aosta", # I changed the name on the df not to deal with special characters
    "Veneto"
]
crime2 = crime1[~crime1['Territory'].isin(regions)]

# get provinces outlines from geojson (https://github.com/openpolis/geojson-italy/blob/master/geojson/limits_IT_provinces.geojson)
provinces = gpd.read_file('limits_IT_provinces.geojson')
#merge crime db with provinces and geospatial coordinates
merged = provinces.merge(crime2, left_on='prov_name', right_on='Territory', how='left')

fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # adjust rows and columns according to desired display
axs = axs.flatten()  # Flatten the 2D array of axes to iterate better in the for loop

crime_categories = [] #insert crimes you want to be plotted

# all crime categories:
        # 'arson', 'attacks', 'attempted homicides', 'bag-snatching',
        # 'bank robbery', 'blows', 'burglary', 'car theft',
        # 'child pornography and possession of paedo-pornographic materials',
        # 'corruption of a minor',
        # 'counterfeiting of goods and industrial products',
        # 'criminal association', 'culpable injuries', 'cybercrime',
        # 'damage followed by arson', 'damages',
        # 'exploitation and abetting prostitution', 'extortions', 'forest arson',
        # 'homicides for theft or robbery', 'homicides from road accident',
        # 'homicides of mafia', 'house robbery', 'infanticides',
        # 'intellectual property violations', 'intentional homicides',
        # 'kidnappings', 'mafia criminal association', 'manslaughter',
        # 'mass murder', 'menaces', 'money laundering', 'moped theft',
        # 'motorcycle theft', 'other crimes', 'pickpocketing',
        # 'post office robbery', 'receiving stolen goods', 'robberies',
        # 'sexual activity with a minor', 'sexual violence', 'shop robbery',
        # 'shoplifting', 'smuggling', 'street robbery',
        # 'swindles and cyber frauds', 'terrorist homicides',
        # 'theft from vehicle', 'theft of art objets',
        # 'theft of cargo trucks carrying freights', 'thefts', 'total',
        # 'trafficking and drugs possession', 'unintentional homicides', 'usury'


# plot each crime on the provinces
for i, category in enumerate(crime_categories):
    merged.plot(column=category, ax=axs[i], legend=True, cmap='viridis',
                missing_kwds={"color": "lightgrey", "label": "Missing values"})
    axs[i].set_title(category.replace('_', ' ').capitalize())
    axs[i].set_axis_off() 

plt.tight_layout()
plt.show()
