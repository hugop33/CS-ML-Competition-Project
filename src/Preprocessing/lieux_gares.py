from config import CITY_PATH
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


def gare_departement(gare):
    return GARES[gare]["departement"]


def gare_region(gare):
    return GARES[gare]["region"]


def city_name(gare_name):
    i = 0
    thresh = 4
    city = ""
    names = gare_name.lower().split(" ")
    while len(city) < thresh:
        city += names[i]
        i += 1
    return city


def distance_intergares(city_csv_path, gare1, gare2):
    city_df = pd.read_csv(city_csv_path, sep=",")
    city1 = city_name(gare1)
    city2 = city_name(gare2)
    # city1_df est le df avec les villes qui contiennent le nom city1
    city1_df = city_df[city_df["label"].str.contains(city1)]
    # city2_df est le df avec les villes qui contiennent le nom city2
    city2_df = city_df[city_df["label"].str.contains(city2)]

    lat1, long1 = city1_df["latitude"].values.mean(
    ), city1_df["longitude"].values.mean()
    lat2, long2 = city2_df["latitude"].values.mean(
    ), city2_df["longitude"].values.mean()

    return geodesic((lat1, long1), (lat2, long2)).km
