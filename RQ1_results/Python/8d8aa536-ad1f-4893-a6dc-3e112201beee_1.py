dfModelSegSummaries = pd.DataFrame([
    [2032, 'data/model-output/v9_RTP_SE32_Net32_Summary_SEGID.csv'],
    [2042, 'data/model-output/v9_RTP_SE42_Net42_Summary_SEGID.csv'],
    [2050, 'data/model-output/v9_RTP_SE50_Net50_Summary_SEGID.csv'],
    [2019, 'data/model-output/v9_SE19_Net19_Summary_SEGID.csv'],
    [2023, 'data/model-output/v9_SE23_Net23_Summary_SEGID.csv'],
    [2028, 'data/model-output/v9_TIP_SE28_Net28_Summary_SEGID.csv'],
], columns=('modelYear', 'modelSegSummaryFile'))

# Create a list to store DataFrames read from each CSV
frames = []

# Iterate through the rows and read each CSV
for index, row in dfModelSegSummaries.iterrows():
    df = pd.read_csv(row['modelSegSummaryFile'])
    df['modelYear'] = row['modelYear'] # Add modelYear column
    frames.append(df)

# Concatenate all the frames into a single DataFrame
result = pd.concat(frames, ignore_index=True)

# You can print or return the resulting DataFrame
print(result)
