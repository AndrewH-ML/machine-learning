import matplotlib.pyplot as plt
import numpy as np

def EWA_PLOT(X, beta, steps, plot_color):
    ewa = [X[0]] 

    for t in range(1, steps):
        ewa_t = beta * ewa[-1] + (1 - beta) * X[t]
        ewa.append(ewa_t)

    plt.plot(ewa, label=f"EWA (beta={beta})", color=plot_color)

np.random.seed(0)
temperature_data = np.random.uniform(65, high=80, size=100)


plt.figure(figsize=(10, 6))
plt.plot(temperature_data, label = "Temperature(t)", color='lightblue')
plt.xlabel("Steps")
plt.ylabel("Temperature (F*)")
plt.title("Exponential Weighted Average")
plt.grid(True)


EWA_PLOT(temperature_data, beta=0.9, steps=100, plot_color='blue')
EWA_PLOT(temperature_data, beta=0.8, steps=100, plot_color='green')
EWA_PLOT(temperature_data, beta=0.5, steps=100, plot_color='purple')

plt.legend(loc="upper right")
plt.show()
