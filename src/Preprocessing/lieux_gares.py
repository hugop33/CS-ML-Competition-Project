from config import STATIONS_PATH
import pandas as pd
from geopy.distance import geodesic

GARES = {
    "BORDEAUX ST JEAN": {
        "region": "NOUVELLE AQUITAINE",
        "departement": "GIRONDE"
    },
    "LA ROCHELLE VILLE": {
        "region": "NOUVELLE AQUITAINE",
        "departement": "CHARENTE MARITIME"
    },
    "PARIS MONTPARNASSE": {
        "region": "ILE DE FRANCE",
        "departement": "PARIS"
    },
    "QUIMPER": {
        "region": "BRETAGNE",
        "departement": "FINISTERE"
    },
    "RENNES": {
        "region": "BRETAGNE",
        "departement": "ILLE ET VILAINE"
    },
    "ST PIERRE DES CORPS": {
        "region": "CENTRE VAL DE LOIRE",
        "departement": "INDRE ET LOIRE"
    },
    "TOURS": {
        "region": "CENTRE VAL DE LOIRE",
        "departement": "INDRE ET LOIRE"
    },
    "NANTES": {
        "region": "PAYS DE LA LOIRE",
        "departement": "LOIRE ATLANTIQUE"
    },
    "PARIS EST": {
        "region": "ILE DE FRANCE",
        "departement": "PARIS"
    },
    "STRASBOURG": {
        "region": "GRAND EST",
        "departement": "BAS RHIN"
    },
    "DUNKERQUE": {
        "region": "HAUTS DE FRANCE",
        "departement": "NORD"
    },
    "LILLE": {
        "region": "HAUTS DE FRANCE",
        "departement": "NORD"
    },
    "PARIS VAUGIRARD": {
        "region": "ILE DE FRANCE",
        "departement": "PARIS"
    },
    "TOURCOING": {
        "region": "HAUTS DE FRANCE",
        "departement": "NORD"
    },
    "CHAMBERY CHALLES LES EAUX": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "SAVOIE"
    },
    "LYON PART DIEU": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "RHONE"
    },
    "MONTPELLIER": {
        "region": "OCCITANIE",
        "departement": "HERAULT"
    },
    "MULHOUSE VILLE": {
        "region": "GRAND EST",
        "departement": "HAUT RHIN"
    },
    "NICE VILLE": {
        "region": "PROVENCE ALPES COTE D'AZUR",
        "departement": "ALPES MARITIMES"
    },
    "PARIS LYON": {
        "region": "ILE DE FRANCE",
        "departement": "PARIS"
    },
    "BARCELONA": {
        "region": "ESPAGNE",
        "departement": "ESPAGNE"
    },
    "GENEVE": {
        "region": "SUISSE",
        "departement": "SUISSE"
    },
    "MADRID": {
        "region": "ESPAGNE",
        "departement": "ESPAGNE"
    },
    "BREST": {
        "region": "BRETAGNE",
        "departement": "FINISTERE"
    },
    "POITIERS": {
        "region": "NOUVELLE AQUITAINE",
        "departement": "VIENNE"
    },
    "TOULOUSE MATABIAU": {
        "region": "OCCITANIE",
        "departement": "HAUTE GARONNE"
    },
    "REIMS": {
        "region": "GRAND EST",
        "departement": "MARNE"
    },
    "DOUAI": {
        "region": "HAUTS DE FRANCE",
        "departement": "NORD"
    },
    "PARIS NORD": {
        "region": "ILE DE FRANCE",
        "departement": "PARIS"
    },
    "BELLEGARDE (AIN)": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "AIN"
    },
    "MACON LOCHE": {
        "region": "BOURGOGNE FRANCHE COMTE",
        "departement": "SAONE ET LOIRE"
    },
    "MARNE LA VALLEE": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "RHONE"
    },
    "PERPIGNAN": {
        "region": "OCCITANIE",
        "departement": "PYRENEES ORIENTALES"
    },
    "TOULON": {
        "region": "PROVENCE ALPES COTE D'AZUR",
        "departement": "VAR"
    },
    "LAUSANNE": {
        "region": "SUISSE",
        "departement": "SUISSE"
    },
    "ANGERS SAINT LAUD": {
        "region": "PAYS DE LA LOIRE",
        "departement": "MAINE ET LOIRE"
    },
    "METZ": {
        "region": "GRAND EST",
        "departement": "MOSELLE"
    },
    "NANCY": {
        "region": "GRAND EST",
        "departement": "MEURTHE ET MOSELLE"
    },
    "BESANCON FRANCHE COMTE TGV": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "RHONE"
    },
    "GRENOBLE": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "ISERE"
    },
    "NIMES": {
        "region": "OCCITANIE",
        "departement": "GARD"
    },
    "SAINT ETIENNE CHATEAUCREUX": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "LOIRE"
    },
    "FRANCFORT": {
        "region": "ALLEMAGNE",
        "departement": "ALLEMAGNE"
    },
    "ITALIE": {
        "region": "ITALIE",
        "departement": "ITALIE"
    },
    "ZURICH": {
        "region": "SUISSE",
        "departement": "SUISSE"
    },
    "ANGOULEME": {
        "region": "NOUVELLE AQUITAINE",
        "departement": "CHARENTE"
    },
    "VANNES": {
        "region": "BRETAGNE",
        "departement": "MORBIHAN"
    },
    "MARSEILLE ST CHARLES": {
        "region": "PROVENCE ALPES COTE D'AZUR",
        "departement": "BOUCHES DU RHONE"
    },
    "ANNECY": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "HAUTE SAVOIE"
    },
    "AVIGNON TGV": {
        "region": "PROVENCE ALPES COTE D'AZUR",
        "departement": "BOUCHES DU RHONE"
    },
    "VALENCE ALIXAN TGV": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "DROME"
    },
    "STUTTGART": {
        "region": "ALLEMAGNE",
        "departement": "ALLEMAGNE"
    },
    "LAVAL": {
        "region": "PAYS DE LA LOIRE",
        "departement": "MAYENNE"
    },
    "LE MANS": {
        "region": "PAYS DE LA LOIRE",
        "departement": "SARTHE"
    },
    "ST MALO": {
        "region": "BRETAGNE",
        "departement": "ILLE ET VILAINE"
    },
    "ARRAS": {
        "region": "HAUTS DE FRANCE",
        "departement": "PAS DE CALAIS"
    },
    "AIX EN PROVENCE TGV": {
        "region": "AUVERGNE RHONE ALPES",
        "departement": "AIN"
    },
    "DIJON VILLE": {
        "region": "GRAND EST",
        "departement": "AUBE"
    },
    "LE CREUSOT MONTCEAU MONTCHANIN": {
        "region": "BOURGOGNE FRANCHE COMTE",
        "departement": "SAONE ET LOIRE"
    }
}

INTERNATIONAL_COORD = {
    "BARCELONA": [41.38227073, 2.177385092],
    "GENEVE": [46.208942, 6.145262],
    "MADRID": [40.416824, -3.703733],
    "FRANCFORT": [50.107149, 8.663785],
    "ITALIE": [45.46663209, 9.190599918],
    "ZURICH": [47.378968, 8.540534],
    "STUTTGART": [48.784081, 9.181636],
    "LAUSANNE": [46.516704, 6.62905],
    "PARIS": [48.8566969, 2.3514616]
}


def gare_departement(gare):
    return GARES[gare]["departement"]


def gare_region(gare):
    return GARES[gare]["region"]


def city_name(gare_name):
    i = 0
    thresh = 4
    city = []
    names = gare_name.lower().split(" ")
    while len("".join(city)) < thresh:
        city.append(names[i])
        i += 1
    return "-".join(city)


def distance_map(city_csv_path, cities):
    city_df = pd.read_csv(city_csv_path, sep=";").dropna(
        subset=["latitude", "longitude"])
    mapping = {}
    for row in cities:
        city1, city2 = row.split("|")
        if city1 in INTERNATIONAL_COORD.keys():
            lat1, long1 = INTERNATIONAL_COORD[city1]
        elif city2 in INTERNATIONAL_COORD.keys():
            lat2, long2 = INTERNATIONAL_COORD[city2]

        else:
            city1_n, city2_n = city_name(city1), city_name(city2)
            # city1_df est le df avec les villes qui contiennent le nom city1
            city1_df = city_df[city_df["slug"].str.startswith(city1_n)]
            # city2_df est le df avec les villes qui contiennent le nom city2
            city2_df = city_df[city_df["slug"].str.startswith(city2_n)]

            lat1, long1 = city1_df["latitude"].values.mean(
            ), city1_df["longitude"].values.mean()
            lat2, long2 = city2_df["latitude"].values.mean(
            ), city2_df["longitude"].values.mean()
        try:
            mapping[(city1, city2)] = geodesic((lat1, long1), (lat2, long2)).km
        except:
            print("Error with cities: ", city1, city2)
            print("lat1, long1: ", lat1, long1)
            print("lat2, long2: ", lat2, long2)
            print("city1_df: ", city1_df[["slug", "latitude", "longitude"]])
            print("city2_df: ", city2_df[["slug", "latitude", "longitude"]])
    return mapping


if __name__ == "__main__":
    print(distance_map(STATIONS_PATH, [
          "PARIS LYON|BESANCON FRANCHE COMTE TGV"]))
