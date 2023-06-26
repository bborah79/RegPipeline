import re


def engineer_feature(df):
    # Convert the Max_torque and Max_power features to numeric features
    df["Engine_f"] = [int(val.split()[0]) for val in list(df["Engine"])]
    df["Engine_hp"] = [float(re.split(r" |@", val)[0]) for val in list(df["Max_power"])]

    df.drop(columns=["Engine", "Max_power", "Max_torque"], inplace=True)
    feature_names = [val for val in df.columns if val != "Price"]

    df["Year"] = df["Year"].apply(int)
    df["Seating_capacity"] = df["Seating_capacity"].apply(str)

    return feature_names, df
