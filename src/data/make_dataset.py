import pandas as pd

TARGET = "violentPerPop"

COLUMN_NAMES = [
    "communityname", "State", "countyCode", "communityCode", "fold",
    "pop", "perHoush", "pctBlack", "pctWhite", "pctAsian", "pctHisp",
    "pct12-21", "pct12-29", "pct16-24", "pct65up", "persUrban", "pctUrban",
    "medIncome", "pctWwage", "pctWfarm", "pctWdiv", "pctWsocsec", "pctPubAsst",
    "pctRetire", "medFamIncome", "perCapInc", "whitePerCap", "blackPerCap",
    "NAperCap", "asianPerCap", "otherPerCap", "hispPerCap", "persPoverty",
    "pctPoverty", "pctLowEdu", "pctNotHSgrad", "pctCollGrad", "pctUnemploy",
    "pctEmploy", "pctEmployMfg", "pctEmployProfServ", "pctOccupManu",
    "pctOccupMgmt", "pctMaleDivorc", "pctMaleNevMar", "pctFemDivorc",
    "pctAllDivorc", "persPerFam", "pct2Par", "pctKids2Par", "pctKids-4w2Par",
    "pct12-17w2Par", "pctWorkMom-6", "pctWorkMom-18", "kidsBornNevrMarr",
    "pctKidsBornNevrMarr", "numForeignBorn", "pctFgnImmig-3", "pctFgnImmig-5",
    "pctFgnImmig-8", "pctFgnImmig-10", "pctImmig-3", "pctImmig-5",
    "pctImmig-8", "pctImmig-10", "pctSpeakOnlyEng", "pctNotSpeakEng",
    "pctLargHousFam", "pctLargHous", "persPerOccupHous", "persPerOwnOccup",
    "persPerRenterOccup", "pctPersOwnOccup", "pctPopDenseHous",
    "pctSmallHousUnits", "medNumBedrm", "houseVacant", "pctHousOccup",
    "pctHousOwnerOccup", "pctVacantBoarded", "pctVacant6up", "medYrHousBuilt",
    "pctHousWOphone", "pctHousWOplumb", "ownHousLowQ", "ownHousMed",
    "ownHousUperQ", "ownHousQrange", "rentLowQ", "rentMed", "rentUpperQ",
    "rentQrange", "medGrossRent", "medRentpctHousInc", "medOwnCostpct",
    "medOwnCostPctWO", "persEmergShelt", "persHomeless", "pctForeignBorn",
    "pctBornStateResid", "pctSameHouse-5", "pctSameCounty-5", "pctSameState-5",
    "numPolice", "policePerPop", "policeField", "policeFieldPerPop",
    "policeCalls", "policCallPerPop", "policCallPerOffic", "policePerPop2",
    "racialMatch", "pctPolicWhite", "pctPolicBlack", "pctPolicHisp",
    "pctPolicAsian", "pctPolicMinority", "officDrugUnits", "numDiffDrugsSeiz",
    "policAveOT", "landArea", "popDensity", "pctUsePubTrans", "policCarsAvail",
    "policOperBudget", "pctPolicPatrol", "gangUnit", "pctOfficDrugUnit",
    "policBudgetPerPop",
    "murders", "murdPerPop", "rapes", "rapesPerPop", "robberies", "robbbPerPop",
    "assaults", "assaultPerPop", "burglaries", "burglPerPop", "larcenies",
    "larcPerPop", "autoTheft", "autoTheftPerPop", "arsons", "arsonsPerPop",
    "violentPerPop", "nonViolPerPop"
]

LEAKAGE_COLS = [
    "murders", "murdPerPop", "rapes", "rapesPerPop", "robberies", "robbbPerPop",
    "assaults", "assaultPerPop", "burglaries", "burglPerPop", "larcenies",
    "larcPerPop", "autoTheft", "autoTheftPerPop", "arsons", "arsonsPerPop",
    "nonViolPerPop"
]

ID_COLS = ["communityname", "countyCode", "communityCode", "fold"]

def load_dataset(path):
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        low_memory=False
    )

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Ocekivano {len(COLUMN_NAMES)} kolona, ucitano {df.shape[1]}. "
            "Proveri fajl/delimiter."
        )

    if TARGET not in df.columns:
        raise ValueError(f"TARGET '{TARGET}' ne postoji u kolonama!")
    
    return df

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=ID_COLS, errors="ignore")
    df = df.drop(columns=LEAKAGE_COLS, errors="ignore")
    return df

def drop_missing_target(df):
    return df.dropna(subset=[TARGET])

def prepare_dataframe(df, missing_threshold=0.80):
    df = df.copy()
    df = drop_columns(df)
    df = drop_missing_target(df)
    return df