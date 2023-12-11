flattened_list = [item for sublist in color_preparation_list for item in (sublist.tolist() if isinstance(sublist, np.ndarray) else sublist)]
