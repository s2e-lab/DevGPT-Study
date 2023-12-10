# ...
# Predict for the specified future years
future_aadt = model.predict(np.array(future_years).reshape(-1, 1))

# Round the forecasted values to the nearest integers
future_aadt = np.rint(future_aadt).astype(int)

# Create a dictionary to store results for this group
result_dict = {'SEGID': segid, 'SOURCE': source, 'Projection Group': pgName}
result_dict.update({year: aadt for year, aadt in zip(future_years, future_aadt)})
# ...
