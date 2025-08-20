# MAC0318 Programação de Robôs Móveis
#
# Preencha o cabeçalho abaixo com seus dados. 
# Se o trabalho for feito em grupo, coloque os nomes de 
# todos os integrantes (copie e cole as linhas abaixo)
#
# Nome:
# NUSP:
#
# ---
#
# Atividade - Veículo de Braitenberg para seguidor de faixa
#
# Objetivo:
#  - Usar a arquitetura de um veículo de Braitenberg para implementar um veículo seguidor de faixa.
#
# Não se esqueça de executar este arquivo do diretório raiz do Duckievillage (p.ex.):
#   cd ~/duckievillage
#   source bin/activate
#   python3 assignments/braitenberg_lf/lane_following.py
#
# Instruções:
#  0. Preencha o cabeçalho acima
#  1. Assegure-se que o código submetido é o correto e funcionando. Se seu código não roda, você receberá nota 0 na atividade.
#  2. Submeta este arquivo no e-disciplinas.

import sys
import pyglet
import numpy as np
from pyglet.window import key
from duckievillage import create_env
import cv2

class Agent:
    # Agent initialization
    def __init__(self, environment):
        """ Initializes agent """
        self.env = environment
        # Color segmentation hyperspace - TODO: MODIFY THE VALUES BELOW
        self.inner_lower = np.array([0, 0, 0])
        self.inner_upper = np.array([179, 255, 255])
        self.outer_lower = np.array([0, 0, 0])
        self.outer_upper = np.array([179, 255, 255])
        # Acquire image for initializing activation matrices
        img = self.env.front()
        img_shape = img.shape[0], img.shape[1]
        self.inner_left_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.inner_right_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.outer_left_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.outer_right_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        # Connecition matrices - TODO: Replace with your code
        self.inner_left_motor_matrix[:, :img_shape[1]//2] = 1
        self.inner_right_motor_matrix[:, img_shape[1]//2:] = 1

    # Image processing routine - Color segmentation
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """ Returns a 2D array mask color segmentation of the image """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # obtain HSV representation of image
        # filter out dashed yellow "inner" line
        inner_mask = cv2.inRange(hsv, self.inner_lower, self.inner_upper)//255
        # filter out solid white "outer" line
        outer_mask = cv2.inRange(hsv, self.outer_lower, self.outer_upper)//255
        # Note: it is possible to filter out pixels in the RGB format
        #  by replacing `hsv` with `image` in the commands above
        # produces combined mask (might or might not be useful)
        mask = cv2.bitwise_or(inner_mask, outer_mask)
        self.masked = cv2.bitwise_and(image, image, mask=mask)
        return inner_mask, outer_mask, mask

    def send_commands(self, dt):
        ''' Agent control loop '''
        # acquire front camera image
        img = self.env.front()
        # run image processing routines
        P, Q, M = self.preprocess(img) # returns inner, outter and combined mask matrices
        # build left and right motor signals from connection matrices and masks (this is a suggestion, feel free to modify it)
        L = float(np.sum(P * self.inner_left_motor_matrix)) + float(np.sum(Q * self.outer_left_motor_matrix))
        R = float(np.sum(P * self.inner_right_motor_matrix)) + float(np.sum(Q * self.outer_right_motor_matrix))
        # Upper bound on the values above (very loose bound)
        limit = img.shape[0]*img.shape[1]*2
        # These are big numbers, better to rescale them to the unit interval
        L = rescale(L, 0, limit)
        R = rescale(R, 0, limit)
        # Tweak with the constants below to get to change velocity or to stabilize the behavior
        # Recall that the pwm signal sets the wheel torque, and is capped to be in [-1,1]
        gain = 3.0   # increasing this will increasing responsitivity and reduce stability
        const = 0.15 # power under null activation - this affects the base velocity
        pwm_left = const + R * gain
        pwm_right = const + L * gain
        # print('>', L, R, pwm_left, pwm_right) # uncomment for debugging
        # Now send command to motors
        self.env.step(pwm_left, pwm_right)
        #  for visualization
        self.env.render('human')


def rescale(x: float, L: float, U: float):
    ''' Map scalar x in interval [L, U] to interval [0, 1]. '''
    return (x - L) / (U - L)


def main():
    print("MAC0318 - Assignment 3")
    env = create_env(
      raw_motor_input = True,
      seed = 101,
      map_name = './maps/loop_empty',
      draw_curve = False,
      draw_bbox = False,
      domain_rand = False,
      #color_sky = [0, 0, 0],
      user_tile_start = (0, 0),
      distortion = False,
      top_down = False,
      cam_height = 10,
      #is_external_map = True,
    )

    angle = env.unwrapped.cam_angle[0]

    env.start_pose = [[0.8, 0, 0.8], 4.5] # initial pose - position and heading
    env.reset()
    env.render('human') # show visualization

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE: # exit simulation
            env.close()
            sys.exit(0)
        elif symbol == key.BACKSPACE or symbol == key.SLASH: # reset simulation
            print("RESET")
            env.reset()
        elif symbol == key.RETURN:  # Take a screenshot
            print('saving screenshot')
            img = env.render('rgb_array')
            cv2.imwrite(f'screenshot-{env.unwrapped.step_count}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'screenshot-masked-{env.unwrapped.step_count}.png', cv2.cvtColor(agent.masked, cv2.COLOR_RGB2BGR))
        env.render() # show image to user

    # Instantiate agent
    agent = Agent(env)
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    # When it's done, close environment and exit program
    env.close()

if __name__ == '__main__':
    main()
