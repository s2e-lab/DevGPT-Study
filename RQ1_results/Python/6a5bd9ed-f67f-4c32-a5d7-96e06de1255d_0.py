elif type_var == 'unit_load_cost':
    data = retrieve_hass.get_attr_data_dict(data_df, idx, entity_id, unit_of_measurement, 
                                            friendly_name, "unit_load_cost_forecasts", state)
elif type_var == 'unit_prod_price':
    data = retrieve_hass.get_attr_data_dict(data_df, idx, entity_id, unit_of_measurement, 
                                            friendly_name, "unit_prod_price_forecasts", state)
