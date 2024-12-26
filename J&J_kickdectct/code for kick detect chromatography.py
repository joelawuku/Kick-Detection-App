import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize figure and axes
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
ax1.set_xlim(0, 5000)  # Initial x-axis limit for plot 1 (Depth)
ax1.set_ylim(0.5, 1.5)  # Initial y-axis limit for plot 1 (Equivalent circulating density - ECD)
ax1.set_xlabel('Depth')
ax1.set_ylabel('S.G.')
ax1.set_title('Equivalent Circulating Density (ECD)')

ax2.set_xlim(0, 5000)  # Initial x-axis limit for plot 2 (Depth)
ax2.set_ylim(20, 40)  # Initial y-axis limit for plot 2 (Formation Temperature)
ax2.set_xlabel('Depth')
ax2.set_ylabel('°C')
ax2.set_title('Formation Temperature')

ax3.set_xlim(0, 5000)  # Initial x-axis limit for plot 3 (Depth)
ax3.set_ylim(50, 100)  # Initial y-axis limit for plot 3 (Bottomed Hole Pressure)
ax3.set_xlabel('Depth')
ax3.set_ylabel('bar')
ax3.set_title('Bottomed Hole Pressure')

# Initialize line plots for all plots
line1, = ax1.plot([], [], label='ECD', color='blue')
line2, = ax2.plot([], [], label='Formation Temperature', color='red')
line3, = ax3.plot([], [], label='Bottomed Hole Pressure', color='green')

# Initialize lists to store previous points for all plots
previous_points1 = []
previous_points2 = []
previous_points3 = []

depth_setpoint = 5000  # Depth setpoint


# Function to update line plot and x-axis for plot 1 (Equivalent circulating density - ECD)
def update_plot1(frame):
    if frame <= depth_setpoint:
        # Generate random Equivalent circulating density (ECD) value
        ecd = np.random.uniform(1.0, 1.5)

        # Add current point to the list of previous points for plot 1 (ECD)
        previous_points1.append([frame, ecd])

        # Update line plot with all previous points for plot 1 (ECD)
        line1.set_data(*zip(*previous_points1))

        # Update x-axis limit dynamically for plot 1 (ECD)
        if frame > 0:
            ax1.set_xlim(frame - 1000, frame + 1000)

        # Update y-axis dynamically for plot 1 (ECD)
        ax1.set_ylim(min(0.5, min(previous_points1, key=lambda x: x[1])[1]) - 0.1,
                     max(1.5, max(previous_points1, key=lambda x: x[1])[1]) + 0.1)

        # Check if frame is equal to depth setpoint
        if frame == depth_setpoint:
            ani1.event_source.stop()  # Stop animation

        return line1,
    else:
        return None

# Function to update line plot and x-axis for plot 2 (Formation Temperature)
def update_plot2(frame):
    if frame <= depth_setpoint:
        # Adjust temperature range based on depth
        if frame <= 2000:
            temp_range = (20, 40)
        else:
            temp_range = (60, 80)

        # Generate random Formation Temperature value within the range
        formation_temp = np.random.uniform(*temp_range)

        # Add current point to the list of previous points for plot 2 (Formation Temperature)
        previous_points2.append([frame, formation_temp])

        # Update line plot with all previous points for plot 2 (Formation Temperature)
        line2.set_data(*zip(*previous_points2))

        # Update x-axis limit dynamically for plot 2 (Formation Temperature)
        if frame > 0:
            ax2.set_xlim(frame - 1000, frame + 1000)

        # Update y-axis dynamically for plot 2 (Formation Temperature)
        ax2.set_ylim(min(20, min(previous_points2, key=lambda x: x[1])[1]) - 5,
                     max(80, max(previous_points2, key=lambda x: x[1])[1]) + 5)

        # Check if frame is equal to depth setpoint
        if frame == depth_setpoint:
            ani2.event_source.stop()  # Stop animation

        return line2,
    else:
        return None

# Function to update line plot and x-axis for plot 3 (Bottomed Hole Pressure)
def update_plot3(frame):
    if frame <= depth_setpoint:
        # Adjust pressure range based on depth
        if frame <= 2000:
            pressure_range = (50, 100)
        else:
            pressure_range = (150, 200)

        # Generate random Bottomed Hole Pressure value within the range
        bhp = np.random.uniform(*pressure_range)

        # Add current point to the list of previous points for plot 3 (Bottomed Hole Pressure)
        previous_points3.append([frame, bhp])

        # Update line plot with all previous points for plot 3 (Bottomed Hole Pressure)
        line3.set_data(*zip(*previous_points3))

        # Update x-axis limit dynamically for plot 3 (Bottomed Hole Pressure)
        if frame > 0:
            ax3.set_xlim(frame - 1000, frame + 1000)

            # Update y-axis dynamically for plot 3 (Bottomed Hole Pressure)
        ax3.set_ylim(min(50, min(previous_points3, key=lambda x: x[1])[1]) - 10,
                     max(200, max(previous_points3, key=lambda x: x[1])[1]) + 10)

        # Check if frame is equal to depth setpoint
        if frame == depth_setpoint:
            ani3.event_source.stop()  # Stop animation

        return line3,
    else:
        return None


# Animate the plots with the specified interval
ani1 = FuncAnimation(fig, update_plot1, frames=np.linspace(0, depth_setpoint, 500), interval=5)
ani2 = FuncAnimation(fig, update_plot2, frames=np.linspace(0, depth_setpoint, 500), interval=5)
ani3 = FuncAnimation(fig, update_plot3, frames=np.linspace(0, depth_setpoint, 500), interval=5)

# Add legend to all plots
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper left')

plt.tight_layout()
plt.show()

