import matplotlib.pyplot as plt
import src.peaks.api as ppi

# dates = ppi.unique_dates()
dataset = ppi.measurement(["2019-05-17 13:42:15"])

plt.plot(dataset["energy"], dataset["count"])
plt.show()
