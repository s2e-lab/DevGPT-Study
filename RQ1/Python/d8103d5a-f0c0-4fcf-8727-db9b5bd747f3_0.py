import matplotlib.pyplot as plt
import numpy as np

def plot_Traceroute(locations):
    # obținerea coordonatelor pentru fiecare locație
    latitudes = []
    longitudes = []
    for location in locations:
        location = geolocator.geocode(location)
        latitudes.append(location.latitude)
        longitudes.append(location.longitude)

    # crearea diagramei
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # plasarea punctelor de pe hartă
    ax.plot(longitudes, latitudes, 'o-', color='blue')

    # adăugarea săgeților între puncte
    for i in range(len(locations)-1):
        start = (longitudes[i], latitudes[i])
        end = (longitudes[i+1], latitudes[i+1])
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        ax.arrow(start[0], start[1], dx, dy, head_width=0.03, head_length=0.03, fc='red', ec='red')

    # setarea limitelor și etichetelor axei
    ax.set_xlim(min(longitudes)-1, max(longitudes)+1)
    ax.set_ylim(min(latitudes)-1, max(latitudes)+1)
    ax.set_xlabel('Longitudine')
    ax.set_ylabel('Latitudine')
    ax.set_title('Traceroute')

    # afișarea diagramei
    plt.show()
