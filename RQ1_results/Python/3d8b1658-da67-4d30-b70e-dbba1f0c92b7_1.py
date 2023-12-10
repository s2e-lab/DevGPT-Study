grouped = df.groupby(['ภาค', 'พรรค'])['คะแนน'].sum().reset_index()
