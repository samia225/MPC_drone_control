import numpy as np
import pygame
import matplotlib.pyplot as plt
from mpc_code import ModelPredictiveControl


def drone():
    A = np.zeros(12)
    B = np.eye(12,4)
    C = np.zeros(12)

    return A,B,C


def keyboard_movements():
    keys = pygame.key.get_pressed()

    #mainly  four movements- thrust,roll, yaw and pitch
    #if no movement occurs we define the state using 0
    thrust = 0
    roll = 0 
    yaw =0 
    pitch = 0


    if keys[pygame.K_UP]:
        thrust = 1 #forcefully increasing thrust

    if keys[pygame.K_DOWN]:
        thrust = -1 #forcefully decreasing thrust

    if keys[pygame.K_RIGHT]:
        roll = 1 #right side dips down, drone moves to the right.

    if keys[pygame.K_LEFT]:
        roll = -1 #left side dips down, drone moves to the left

    if keys[pygame.K_d]:
        yaw = 1 #yaw to the right

    if keys[pygame.K_a]:
        yaw = -1 #yaw to the left

    if keys[pygame.K_w]:
        pitch = 1 #pitch forward - front of the drone tilts downward, and the rear tilts upward

    if keys[pygame.K_s]:
        pitch = -1 #pitch backward-  rear of the drone tilts downward, and the front tilts upward

    return np.array([(thrust),(roll),(yaw),(pitch)])


def initialize_pygame():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption('Drone Control')
    return screen


def mouse_movements():
    # Get mouse position for continuous control
    pos = pygame.mouse.get_pos()
    
    # Normalize between -1 and 1 for x and y movement
    x = 2.0 * (pos[0] - 320) / 640  # scales it between -1, 1 for x
    y = 2.0 * (480 - pos[1] - 240) / 480  # scales it between -1, 1 for y
    
    # Mouse control applied to thrust and pitch (for example)
    thrust = y
    roll = x
    yaw = 0  # could be added later
    pitch = 0  # could be added later
    
    return np.array([thrust, roll, yaw, pitch])


class DroneControl:
    def __init__(self):
        self._A, self._B, self._C = drone()

    def simulation_keyboard(self):
        alpha = 0.5  # 50% human control, 50% autonomous control
        time_steps = 300
        f = 20
        v = 20

        W3 = np.eye(v * 4)  # Input weight matrix (for 4 control inputs)
        W4 = np.eye(f * 12) * 10

        x0_test = np.zeros(shape=(12, 1))
        x0 = x0_test

        desired_trajectory = np.zeros(shape=(time_steps, 12))

        mpc = ModelPredictiveControl(self._A, self._B, self._C, f, v, W3, W4, x0, desired_trajectory)

        # Initialize pygame window
        screen = initialize_pygame()

        controlled_trajectory_list = []
        control_input_list = []

        for i in range(time_steps - f):
            # Capture keyboard movements
            input_h = keyboard_movements()

            # Compute MPC control inputs
            mpc_r = mpc.computeControlInputs()

            # Blend human control with MPC
            pie_augmented = alpha * input_h + (1 - alpha) * mpc_r

            # Log for plotting or further analysis
            control_input_list.append(pie_augmented)

            # Pygame event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Update display (if you want to visualize something)
            pygame.display.update()
