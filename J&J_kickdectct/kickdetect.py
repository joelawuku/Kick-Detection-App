import csv
import sys
from random import random
import random
import mplcursors
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QIcon
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, \
    QWidget, QTextEdit, QFrame, QScrollArea, QAction, QMessageBox, QComboBox
from PyQt5.QtCore import QTimer
from scipy.interpolate import Akima1DInterpolator


class KickDetectApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window icon
        self.setWindowIcon(QIcon('jackup.jpg'))

        # Set fixed window size
        self.setFixedSize(850, 500)  # Adjust width and height as needed

        # Initialize the attributes
        self.ecd_value = 0
        self.temp_value = 0
        self.bhp_value = 0

        self.setWindowTitle("KickDetect")
        self.setGeometry(100, 100, 800, 600)  # Adjusted window size

        self.setupUI()

    def setupUI(self):

        # Create menu bar
        self.createMenuBar()

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)

        layout = QVBoxLayout(centralWidget)

        # Create a horizontal layout for input and dashboard
        inputDashboardLayout = QHBoxLayout()

        # Create input section within a scroll area
        inputScrollArea = QScrollArea()
        inputScrollArea.setWidgetResizable(True)
        inputScrollArea.setFixedWidth(300)
        inputFrame = self.createInputSection()

        inputScrollArea.setWidget(inputFrame)

        # Add input scroll area to the horizontal layout
        inputDashboardLayout.addWidget(inputScrollArea)



        # Create dashboard frame
        dashboardFrame = self.createDashboardFrame()

        # Add dashboard frame to the horizontal layout
        inputDashboardLayout.addWidget(dashboardFrame)

        # Add the horizontal layout (input section and dashboard) to the main vertical layout
        layout.addLayout(inputDashboardLayout)

        # Create status frame for predictions
        statusFrame = self.createStatusFrame()

        # Create and add the image frame
        image_frame = self.create_image_frame()
        inputDashboardLayout.addWidget(image_frame)

        # Add status frame to the main vertical layout
        layout.addWidget(statusFrame)

        # Create control frame for buttons
        controlFrame = self.createControlFrame()

        # Add control frame to the main vertical layout
        layout.addWidget(controlFrame)

        self.startTimer = QTimer()  # Timer for updating LED labels
        self.startTimer.timeout.connect(self.updateLEDLabels)

    def createMenuBar(self):
        menuBar = self.menuBar()

        # File menu
        fileMenu = menuBar.addMenu('File')
        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        # Options menu
        optionsMenu = menuBar.addMenu('Options')
        importAction = QAction('Import', self)
        importAction.triggered.connect(self.importData)


        exportAction = QAction('Export', self)
        exportAction.triggered.connect(self.displayDataInTextWindow)
        optionsMenu.addAction(importAction)
        optionsMenu.addAction(exportAction)

        # About menu
        aboutMenu = menuBar.addMenu('About')
        aboutAction = QAction('About', self)
        aboutAction.triggered.connect(self.showAboutMessage)
        aboutMenu.addAction(aboutAction)

    def importData(self):
        # Data to import
        total_depth = "3000"
        avg_pore_pressure = "800"
        avg_fracture_pressure = "1200"
        formation_strength = "500"
        friction_angle = "45"
        num_targets = "4"
        kop_x = "0"
        kop_y = "0"
        tvd_kop = "800"
        target_coordinates = "238.21,137.53,1164.06\n" \
                             "337.47,194.84,1223.73\n" \
                             "759.59,438.55,1327.21\n" \
                             "1284.59,741.66,1677.21"  # Example target coordinates



        # Populate line edits with imported data
        self.total_depth_edit.setText(total_depth)  # Total Depth
        self.avg_pore_pressure_edit.setText(avg_pore_pressure)  # Average Pore Pressure
        self.avg_fracture_pressure_edit.setText(avg_fracture_pressure)  # Average Fracture Pressure
        self.formation_strength_edit.setText(formation_strength)  # Formation Strength
        self.friction_angle_combo.setCurrentText(friction_angle)  # Friction Angle
        self.num_targets_edit.setText(num_targets)  # Number of Targets
        self.kop_x_edit.setText(kop_x)  # KOP X
        self.kop_y_edit.setText(kop_y)  # KOP Y
        self.tvd_kop_edit.setText(tvd_kop)  # TVD KOP
        self.target_coords_edit.setPlainText(target_coordinates)  # Target Coordinates

    def showAboutMessage(self):
        QMessageBox.about(self, "About", "This app is a development app created by Joel and Jane. 2024. Version 1.0")

    def createControlFrame(self):
        controlFrame = QFrame()
        controlFrame.setFrameShape(QFrame.StyledPanel)
        controlLayout = QHBoxLayout(controlFrame)

        # Create buttons for start, stop, and clear
        startButton = QPushButton("Start")
        startButton.setStyleSheet("background-color: green; color: white;")
        stopButton = QPushButton("Stop")
        stopButton.setStyleSheet("background-color: red; color: white;")
        clearButton = QPushButton("Clear")
        clearButton.setStyleSheet("background-color: blue; color: white;")

        # Connect buttons to appropriate slots
        startButton.clicked.connect(self.startAnimation)
        stopButton.clicked.connect(self.stopAnimation)
        clearButton.clicked.connect(self.clearPlots)

        # Add buttons to the control frame layout
        controlLayout.addWidget(startButton)
        controlLayout.addWidget(stopButton)
        controlLayout.addWidget(clearButton)

        return controlFrame

    def create_image_frame(self):
        image_frame = QFrame()
        image_layout = QVBoxLayout(image_frame)

        self.image_label = QLabel()
        pixmap = QPixmap("kick.jpg")
        pixmap_resized = pixmap.scaled(300, 300)  # Adjust width and height as needed
        self.image_label.setPixmap(pixmap_resized)


        # Adding the image label to the layout
        image_layout.addWidget(self.image_label)

        # Adding the statement label
        statement_label = QLabel(
            "This app is a project work for study purposes. "
            "It does not guarantee 100% accuracy in detecting kicks and should not be relied upon for real-life "
            "situations. The detection algorithms are experimental and may not perform accurately in all complex scenarios."
        )
        statement_label.setWordWrap(True)  # Enable word wrapping for long text
        image_layout.addWidget(statement_label)

        return image_frame

    def createDashboardFrame(self):
        dashboardFrame = QFrame()
        dashboardFrame.setFrameShape(QFrame.StyledPanel)
        dashboardLayout = QVBoxLayout(dashboardFrame)

        # Dashboard labels
        dashboardLabelsLayout = QHBoxLayout()

        ecdLabel = QLabel("ECD (ppg)")
        ecdLabel.setStyleSheet("font-weight: bold")
        tempLabel = QLabel("Temp (°C)")
        tempLabel.setStyleSheet("font-weight: bold")
        bhpLabel = QLabel("BHP (psi)")
        bhpLabel.setStyleSheet("font-weight: bold")

        dashboardLabelsLayout.addWidget(ecdLabel)
        dashboardLabelsLayout.addWidget(tempLabel)
        dashboardLabelsLayout.addWidget(bhpLabel)

        # LED labels
        ledLabelsLayout = QHBoxLayout()

        self.ledECD = QLabel("0")
        self.ledECD.setStyleSheet("color: green;font-size: 14px")  # Set text color to green
        self.ledTemp = QLabel("0")
        self.ledTemp.setStyleSheet("color: green;font-size: 14px")  # Set text color to green
        self.ledBHP = QLabel("0")
        self.ledBHP.setStyleSheet("color: green;font-size: 14px")  # Set text color to green

        ledLabelsLayout.addWidget(self.ledECD)
        ledLabelsLayout.addWidget(self.ledTemp)
        ledLabelsLayout.addWidget(self.ledBHP)

        dashboardLayout.addLayout(dashboardLabelsLayout)
        dashboardLayout.addLayout(ledLabelsLayout)

        return dashboardFrame


    def updateLEDLabels(self):
        # Update LED labels with random values between 0 and 1000
        ecd_value = np.random.randint(0, 1001)
        temp_value = np.random.randint(0, 1001)
        bhp_value = np.random.randint(0, 1001)

        self.ledECD.setText(str(ecd_value))
        self.ledTemp.setText(str(temp_value))
        self.ledBHP.setText(str(bhp_value))

        # Store the values for use in updateStatusLabel
        self.ecd_value = ecd_value
        self.temp_value = temp_value
        self.bhp_value = bhp_value

    def startAnimation(self):
        # Check input validity before starting animation
        if not self.checkInputValidity():
            QMessageBox.critical(self, "Error", "Please enter valid inputs.")
            return

        # Start simulation and plot kick detection
        self.start_simulation()
        self.plot_kick_detection()

        # Start timer to update LED labels every second
        self.startTimer.start(1000)

        # Start timer to stop LED updates after 30 seconds
        self.ledUpdateTimer = QTimer()
        self.ledUpdateTimer.setSingleShot(True)
        self.ledUpdateTimer.timeout.connect(self.stopLEDUpdates)
        self.ledUpdateTimer.start(35000)  # 30 seconds in milliseconds

        # Update status label
        self.updateStatusLabel()

    def stopLEDUpdates(self):
        # Stop QTimer for updating LED labels
        self.startTimer.stop()

    def stopAnimation(self):
        # Stop QTimer for updating LED labels
        self.startTimer.stop()

    def clearPlots(self):
        # Clear input fields
        for lineEdit in self.findChildren(QLineEdit):
            lineEdit.clear()

        # Reset LED labels
        self.ledECD.setText("0")
        self.ledTemp.setText("0")
        self.ledBHP.setText("0")

        # Reset the prediction label
        self.prediction_label.setText("No predictions at this time")
        self.prediction_label.setStyleSheet("color: black;")  # Optional: Reset the color to black

    def createInputSection(self):
        inputFrame = QFrame()
        inputFrame.setFrameShape(QFrame.StyledPanel)
        inputLayout = QVBoxLayout(inputFrame)

        # Label for the input frame
        inputLabel = QLabel("Input")
        inputLabel.setStyleSheet("font-weight: bold; font-size: 14px;")
        inputLayout.addWidget(inputLabel)

        # Formation Data Section
        inputLayout.addWidget(QLabel("Formation Data"))

        # Total Depth
        total_depth_label = QLabel("Total Depth (TVD) (m)")
        self.total_depth_edit = QLineEdit()
        inputLayout.addWidget(total_depth_label)
        inputLayout.addWidget(self.total_depth_edit)

        # Average Pore Pressure
        avg_pore_pressure_label = QLabel("Average Pore Pressure (psi/ft)")
        self.avg_pore_pressure_edit = QLineEdit()
        inputLayout.addWidget(avg_pore_pressure_label)
        inputLayout.addWidget(self.avg_pore_pressure_edit)

        # Average Fracture Pressure
        avg_fracture_pressure_label = QLabel("Average Fracture Pressure (psi)")
        self.avg_fracture_pressure_edit = QLineEdit()
        inputLayout.addWidget(avg_fracture_pressure_label)
        inputLayout.addWidget(self.avg_fracture_pressure_edit)

        # Formation Strength
        formation_strength_label = QLabel("Formation Strength (MPa)")
        self.formation_strength_edit = QLineEdit()
        inputLayout.addWidget(formation_strength_label)
        inputLayout.addWidget(self.formation_strength_edit)

        # Friction Angle
        friction_angle_label = QLabel("Friction Angle (°)")
        self.friction_angle_combo = QComboBox()
        self.friction_angle_combo.addItems(['20', '30'])  # Example options
        inputLayout.addWidget(friction_angle_label)
        inputLayout.addWidget(self.friction_angle_combo)

        # Well Input Section
        inputLayout.addWidget(QLabel("Well Input"))

        self.num_targets_label = QLabel("Number of Targets:")
        self.num_targets_edit = QLineEdit()
        self.kop_x_label = QLabel("KOP X:")
        self.kop_x_edit = QLineEdit()
        self.kop_y_label = QLabel("KOP Y:")
        self.kop_y_edit = QLineEdit()
        self.tvd_kop_label = QLabel("TVD KOP:")
        self.tvd_kop_edit = QLineEdit()
        self.target_coords_label = QLabel("Target Coordinates (x,y,z):")
        self.target_coords_edit = QTextEdit()

        inputLayout.addWidget(self.num_targets_label)
        inputLayout.addWidget(self.num_targets_edit)
        inputLayout.addWidget(self.kop_x_label)
        inputLayout.addWidget(self.kop_x_edit)
        inputLayout.addWidget(self.kop_y_label)
        inputLayout.addWidget(self.kop_y_edit)
        inputLayout.addWidget(self.tvd_kop_label)
        inputLayout.addWidget(self.tvd_kop_edit)
        inputLayout.addWidget(self.target_coords_label)
        inputLayout.addWidget(self.target_coords_edit)

        return inputFrame

    def createStatusFrame(self):
        statusFrame = QFrame()
        statusFrame.setFrameShape(QFrame.StyledPanel)
        self.statusLayout = QVBoxLayout(statusFrame)

        statusLabel = QLabel("Kick Predictions")
        statusLabel.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.statusLayout.addWidget(statusLabel)

        self.prediction_label = QLabel("No predictions at this time")
        self.statusLayout.addWidget(self.prediction_label)

        return statusFrame

    def updateStatusLabel(self):
        try:
            # Use the stored ecd, temp, and bhp values
            ecd = self.ecd_value
            temp = self.temp_value
            bhp = self.bhp_value

            # Example logic for updating the status based on ecd, temp, and bhp values
            if ecd < 300 and temp < 300 and bhp < 300:
                current_status = "NO KICK"
            elif ecd > 700 or temp > 700 or bhp > 700:
                current_status = "KICK"
            else:
                current_status = "ABNORMAL PRESSURES"

            # Update the label text and color based on the current status
            if current_status == "NO KICK":
                self.prediction_label.setText("NO KICK")
                self.prediction_label.setStyleSheet("color: green;")
            elif current_status == "KICK":
                self.prediction_label.setText("KICK")
                self.prediction_label.setStyleSheet("color: red;")
            elif current_status == "ABNORMAL PRESSURES":
                self.prediction_label.setText("ABNORMAL PRESSURES")
                self.prediction_label.setStyleSheet("color: yellow;")
        except Exception as e:
            print(f"An error occurred in updateStatusLabel: {e}")

    def checkInputValidity(self):
        for lineEdit in self.findChildren(QLineEdit):
            if not lineEdit.text():
                return False
        return True

    def plot_kick_detection(self):
        # Initialize figure and axes
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4))  # Adjust the figsize as needed
        fig.canvas.manager.set_window_title("Kick Detection visuals")



        ax1.set_xlim(0, 5000)  # Initial x-axis limit for plot 1 (Depth)
        ax1.set_ylim(0.5, 1.5)  # Initial y-axis limit for plot 1 (Equivalent circulating density - ECD)
        ax1.set_ylabel('S.G.')

        ax2.set_xlim(0, 5000)  # Initial x-axis limit for plot 2 (Depth)
        ax2.set_ylim(20, 40)  # Initial y-axis limit for plot 2 (Formation Temperature)
        ax2.set_ylabel('°C')

        ax3.set_xlim(0, 5000)  # Initial x-axis limit for plot 3 (Depth)
        ax3.set_ylim(50, 100)  # Initial y-axis limit for plot 3 (Bottomed Hole Pressure)
        ax3.set_xlabel('Depth')
        ax3.set_ylabel('bar')

        # Initialize line plots for all plots
        line1, = ax1.plot([], [], label='ECD', color='blue')
        line2, = ax2.plot([], [], label='Formation Temperature', color='red')
        line3, = ax3.plot([], [], label='Bottomed Hole Pressure', color='green')

        # Initialize lists to store previous points for all plots
        previous_points1 = []
        previous_points2 = []
        previous_points3 = []

        depth_setpoint = 2500  # Depth setpoint

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
                    self.ani1.event_source.stop()  # Stop animation

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
                    self.ani2.event_source.stop()  # Stop animation

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
                    self.ani3.event_source.stop()  # Stop animation

                return line3,
            else:
                return None

        # Create animations for each plot
        # Animate the plots with the specified interval
        self.ani1 = FuncAnimation(fig, update_plot1, frames=np.linspace(0, depth_setpoint, 500), interval=5)
        self.ani2 = FuncAnimation(fig, update_plot2, frames=np.linspace(0, depth_setpoint, 500), interval=5)
        self.ani3 = FuncAnimation(fig, update_plot3, frames=np.linspace(0, depth_setpoint, 500), interval=5)

        plt.show()

    def start_simulation(self):
        # Retrieve input values
        num_targets = int(self.num_targets_edit.text())
        kop_x = float(self.kop_x_edit.text())
        kop_y = float(self.kop_y_edit.text())
        tvd_kop = float(self.tvd_kop_edit.text())

        target_coords_text = self.target_coords_edit.toPlainText()
        target_coords = []
        lines = target_coords_text.split("\n")
        for line in lines:
            coordinates = line.strip()
            if coordinates:
                x, y, z = map(float, coordinates.split(","))
                target_coords.append([x, y, z])
        target_coords = np.array(target_coords)
        if len(target_coords) != num_targets:
            self.show_error_message(f"Number of target coordinates does not match the specified number of targets "
                                    f"({num_targets}). Please enter {num_targets} target coordinates.")
            return

        # Surface coordinates
        surface_coords = np.array([0, 0, 0])

        surface_x, surface_y, surface_z = surface_coords
        kop_z = tvd_kop

        targets_x = target_coords[:, 0]
        targets_y = target_coords[:, 1]
        targets_z = target_coords[:, 2]

        interpolating_x = np.insert(targets_x, 0, kop_x)
        interpolating_y = np.insert(targets_y, 0, kop_y)
        interpolating_z = np.insert(targets_z, 0, kop_z)

        def akima1DInterp(x, y, z):
            interp_func_x = Akima1DInterpolator(z, x)
            interp_func_y = Akima1DInterpolator(z, y)
            interp_func_z = Akima1DInterpolator(z, z)

            z_interp = np.arange(z[0], z[-1], 10)
            x_interp = interp_func_x(z_interp)
            y_interp = interp_func_y(z_interp)
            z_interp = interp_func_z(z_interp)

            return x_interp, y_interp, z_interp

        x, y, z = akima1DInterp(interpolating_x, interpolating_y, interpolating_z)

        surface_to_kop_zs = np.arange(surface_z, kop_z, 10)
        z = np.concatenate((surface_to_kop_zs, z))
        x = np.concatenate((np.full((len(surface_to_kop_zs),), surface_x), x))
        y = np.concatenate((np.full((len(surface_to_kop_zs),), surface_y), y))

        # Create a new figure and subplot with 3D projection (animated)
        self.fig_animated = plt.figure(figsize=(4, 4), facecolor='white')
        ax_animated = self.fig_animated.add_subplot(111, projection='3d')

        # Move the matplotlib window to the left side of the screen
        manager = plt.get_current_fig_manager()
        manager.window.move(0, 0)  # (0, 0) moves the window to the top-left corner

        # Set labels for the axes
        ax_animated.set_xlabel('EAST')
        ax_animated.set_ylabel('NORTH')
        ax_animated.set_zlabel('TVD')

        # Invert the z-axis (TVD) for better visualization
        ax_animated.invert_zaxis()

        # Set the title
        plt.title('Drilling Simulation')

        # Set black background for the plot
        ax_animated.set_facecolor('white')

        #  gridlines
        ax_animated.grid(True)

        # Set the window title
        self.fig_animated.canvas.manager.set_window_title('Drilling Simulation')

        # Plot the rig location as a big black triangle marker
        rig_marker = ax_animated.scatter([0], [0], [0], c='Orange', marker='v', s=200,
                                         label='Rig Position')

        # Plot the targets
        target_markers = ax_animated.scatter(target_coords[:, 0], target_coords[:, 1], target_coords[:, 2],
                                             c='r', marker='o', label='Core targets')

        well_plan = ax_animated.plot(x, y, z, c='c', label='well plan')

        # Initialize the line for the simulated well path
        line_deviated, = ax_animated.plot([], [], [], c='black', label='Simulated Path', marker='')

        # Kick prediction data
        # Kick prediction data
        kick_labels = [
            (238.21, 137.53, 1164.06, "KICK", 'red'),
            (337.47, 194.84, 1223.73, "NO KICK", 'green'),
            (759.59, 438.55, 1327.21, "KICK", 'red'),
            (1284.59, 741.66, 1677.21, "NO KICK", 'green'),
        ]

        # Add the labels to the plot
        for x_label, y_label, z_label, label, color in kick_labels:
            ax_animated.text(x_label, y_label, z_label, label, color=color, fontsize=10, fontweight='bold')

        # Function to update the animated plot with drift effect
        def update(frame):
            # Add noise to the coordinates for wobbly effect
            noise = np.random.normal(0, 2, size=(len(x), 2))
            x_wobbly = x + noise[:, 0]
            y_wobbly = y + noise[:, 1]

            # Introduce drift in x and y coordinates
            drift_x = np.linspace(0, 0.5, len(x)) * frame * 0.3  # Adjust the drift intensity as needed
            drift_y = np.linspace(0, 0.5, len(y)) * frame * 0.3

            x_drifted = x_wobbly + drift_x
            y_drifted = y_wobbly + drift_y
            z_drifted = z

            # Update the line data with drift effect
            line_deviated.set_data(x_drifted[:frame], y_drifted[:frame])
            line_deviated.set_3d_properties(z_drifted[:frame])

            # Set line properties: big size and transparency
            line_deviated.set_linewidth(3)
            line_deviated.set_alpha(0.5)

            return line_deviated,

        # Create the animation
        num_frames = len(x)
        self.ani = FuncAnimation(self.fig_animated, update, frames=num_frames, interval=100, blit=True, repeat=False)

        # Add a legend to the plot
        ax_animated.legend()

        # Show the animation
        plt.show()

        # Register cursor-based annotations
        mplcursors.cursor(hover=True)

    def displayDataInTextWindow(self):
        # Define the number of samples
        num_samples = 100

        # Define the CSV file path
        csv_file = "kick_detect_data.csv"

        # Open CSV file for writing
        with open(csv_file, mode='w', newline='') as file:
            # Define CSV writer
            fieldnames = [
                "depth", "dip", "azimuth",
                "E", "N", "frac_pressure",
                "pore_pressure", "ECD", "BHP",
                "hole_temp"
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            # Initialize starting values for increasing parameters
            depth_start = 0
            dip_start = 0
            azimuth_start = 0
            pore_pressure_start = 0

            # Generate and write each sample to CSV
            for i in range(num_samples):
                depth = depth_start + i * 50  # Increase depth uniformly
                dip = dip_start + i * 0.5  # Increase dip uniformly
                azimuth = azimuth_start + i * 5  # Increase azimuth uniformly
                pore_pressure = pore_pressure_start + i * 20  # Increase pore pressure uniformly

                # Generate random values for other parameters
                E = random.uniform(0.5, 2.0)
                N = random.uniform(0.5, 2.0)
                frac_pressure = random.uniform(6000, 8000)
                ECD = random.uniform(10.0, 14.0)
                BHP = random.uniform(5000, 7000)
                hole_temp = random.uniform(80, 120)

                data = {
                    "depth": depth,
                    "dip": dip,
                    "azimuth": azimuth,
                    "E": E,
                    "N": N,
                    "frac_pressure": frac_pressure,
                    "pore_pressure": pore_pressure,
                    "ECD": ECD,
                    "BHP": BHP,
                    "hole_temp": hole_temp
                }

                writer.writerow(data)

        # Show pop-up window confirming export
        self.showExportConfirmation()

    def showExportConfirmation(self):
        msg = QMessageBox()
        msg.setWindowTitle("Export Complete")
        msg.setText("Data exported to CSV.")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Set the application style to Fusion
    window = KickDetectApp()
    window.show()
    sys.exit(app.exec_())

