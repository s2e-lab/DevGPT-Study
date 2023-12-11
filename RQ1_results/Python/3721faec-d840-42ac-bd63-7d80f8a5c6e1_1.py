import pandas as pd

path_to_csv_files = "/content/drive/MyDrive/path_to_csv_files/"

names_dates = pd.read_csv(path_to_csv_files + "Names_and_Dates_of_Life.csv")
exile_life_stages = pd.read_csv(path_to_csv_files + "Exile_and_Life_Stages.csv")
professions_activities = pd.read_csv(path_to_csv_files + "Professions_and_Activities.csv")
