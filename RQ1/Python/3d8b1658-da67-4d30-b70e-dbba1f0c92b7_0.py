# List the column names in your DataFrame
print(df.columns)

# Replace "zone" and "name" with the actual column names
party_scores = df.groupby(["ActualZoneColumnName", "ActualNameColumnName"]).sum()["party_list_vote"].reset_index()
party_scores_sorted = party_scores.sort_values(by=["ActualZoneColumnName", "party_list_vote"], ascending=[True, False])
print(party_scores_sorted)
