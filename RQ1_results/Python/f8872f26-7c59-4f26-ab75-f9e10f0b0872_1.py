import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Angenommen, dies ist Ihr DataFrame
data = {'Age': [23, 25, 22, 30, 25, 23, 25, 30, 30, 22, 25]}
df = pd.DataFrame(data)

# Berechne die Anzahl jeder Altersgruppe
age_counts = df['Age'].value_counts().sort_index()

# Stil festlegen
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)

# Barplot erstellen
sns.barplot(x=age_counts.index, y=age_counts.values)

# Diagramm anzeigen
plt.show()
