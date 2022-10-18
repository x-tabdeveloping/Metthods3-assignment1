# %%
import pandas as pd

# %%
demo = pd.read_csv("demo_test.csv")
lu = pd.read_csv("LU_test.csv")
demo.info()
lu.info()
# %%
demo = demo.rename(
    columns={
        "Child.ID": "child_id",
        "Visit": "visit",
        "Diagnosis": "diagnosis",
    }
)[["child_id", "visit", "diagnosis"]]
lu = lu.rename(
    columns={
        "SUBJ": "child_id",
        "VISIT": "visit",
        "CHI_MLU": "child_mlu",
    }
)[["child_id", "visit", "MOT_MLU", "child_mlu"]]
demo = demo.assign(child_id=demo.child_id.str.strip("."))
lu = lu.assign(child_id=lu.child_id.str.strip("."))
lu = lu.assign(visit=lu.visit.str.slice(start=-2, stop=-1).astype(int))
#%%
# %%
data = demo.merge(lu, on=["visit", "child_id"], how="inner")
data = data.dropna(
    axis="index",
    subset=["visit", "child_id", "child_mlu", "MOT_MLU", "diagnosis"],
)
data = data[data.child_mlu != 0]
# Mapping diagnoses to their ID
data = data.assign(diagnosis=data.diagnosis.map({"A": 1, "B": 0}))
# Factorizing children
id_values, id_uniques = pd.factorize(data.child_id)
data["child_id"] = id_values
data["visit"] = data["visit"] - 1
data.to_csv("test_clean.csv")
