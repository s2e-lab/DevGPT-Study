# Set all columns to the specified values
df_melted["economy"] = "19_THA"
df_melted["date"] = 2020
df_melted["medium"] = "road"
df_melted["measure"] = "stocks"
df_melted["dataset"] = "9th_model_manual_updates"
df_melted["source"] = "thanan"
df_melted["unit"] = "stocks"
df_melted["fuel"] = "all"
df_melted["comment"] = "no_comment"
df_melted["scope"] = "national"
df_melted["frequency"] = "yearly"

# Save the final DataFrame to an Excel file
df_melted.to_excel("/mnt/data/chatgpt_stocks_output_with_constant_values.xlsx", index=False)
