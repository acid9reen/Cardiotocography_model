import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import elpigraph


DATA_PATH = r"./data/CTG.xls"


def load_cardiotography(filepath: str):
    return pd.read_excel(filepath, sheet_name="Raw Data")


cardiotography = load_cardiotography(DATA_PATH)
class_labels = ["A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", "SUSP"]

ad_columns_to_drop = ["e", "LBE", "Mean"]

cardiotography = cardiotography.drop(
    index=[0, 2127, 2128, 2129],
    columns=["FileName", "Date", "SegFile", "NSP"] + class_labels + [],
)

cardiotography.CLASS[:] = [class_labels[int(i) - 1] for i in cardiotography.CLASS]


num_attribs = [attr for attr in cardiotography.keys() if attr != "CLASS"]

num_pipeline = Pipeline(
    [
        ("std_scalar", StandardScaler()),
    ]
)

full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
    ]
)

cardiotography_prep = full_pipeline.fit_transform(cardiotography)

PG = elpigraph.computeElasticPrincipalTree(cardiotography_prep, 200)[0]
elpigraph.plot.PlotPG(cardiotography_prep, PG)
