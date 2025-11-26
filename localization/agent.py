# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Iago Cesar Tavares de Souza
# NUSP: 17466770
#
# ---
#
# Assignment 11 - Lane-following localization estimation
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/localization/agent.py
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change hasn't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file via e-disciplinas.

import sys
import pyglet
import numpy as np
import math
import random
from pyglet.window import key
from duckievillage import create_env, FRONT_VIEW_MODE, Histogram
from scipy.stats import norm as gaussian
import cv2
import tensorflow

# Uncomment the following line if you get an error with the parallel library
# accusing multiple instantiations
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Agent:
    def __init__(self, env):
        self.env = env
        self.radius = 0.0318 # R
        self.baseline = env.unwrapped.wheel_dist/2
        self.motor_gain = 0.68*0.0784739898632288
        self.motor_trim = 0.0007500911693361842

        self.pose_estimator = Agent.load_regression_model("./assignments/localization/img/regression-model.h5")
        self.score = 0

        self.C = 6
        self.I = 0.0
        self.D = 0.0

        self.Kp, self.Kd, self.Ki = 2, 0.5, 0
        self.integralError = 0
        self.previousError = 0
        self.rotation = 0

        # Get test set states (d, alpha).
        Y = np.load("./assignments/localization/img/test-labels.npy", allow_pickle = True)
        # Linearize.
        Y_t = 6*Y
        # Standard deviation for the transition Gaussians.
        self.transition_sigma = 0.7

        # Discretize y's into cells.
        n = 10
        self.cells = Agent.discretize(np.min(Y_t), np.max(Y_t)+1e-10, n)
        self.gaussians = Agent.create_gauss(self.cells, Y_t)

        # Current belief.
        self.bel = np.ones(n)/n
        # Auxiliary variable for computing probabilities.
        self.pr = np.zeros(n)

        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

        self.hist = Histogram(self.bel, self.cells)

    @staticmethod
    def discretize(low: float, high: float, n: int) -> list:
        ''' Discretizes a set of reals [low, high) into n cells, returning a list of each disjoint
        interval I_i = [a_i, b_i), for all i in 0..n-1 st the union of all I_i is [low, high). '''
        k = (high-low)/n
        return [(low+k*i, low+k*(i+1)) for i in range(n)]

    @staticmethod
    def create_gauss(cells: list, Y_t: np.ndarray) -> list:
        ''' Creates |cells| Gaussians with mean and variance estimated from Y_t. Each Gaussian is
        restricted to a cell's range and so only estimates from values in that range. '''
        Y_t.sort()
        w, l = 0, 0
        S = [None for _ in cells]
        for i, y in enumerate(Y_t):
            first = True
            a, b = cells[w]
            while y > b:
                if first:
                    sigma = np.std(Y_t[l:i], ddof = 1)
                    if math.isnan(sigma): sigma = 1e-2
                    S[w] = (np.mean(Y_t[l:i]), sigma)
                w += 1
                l, first = i, False
                a, b = cells[w]
        sigma = np.std(Y_t[l:], ddof = 1)
        if math.isnan(sigma): sigma = 1e-2
        S[w] = (np.mean(Y_t[l:]), sigma)
        return S

    @staticmethod
    def linearize(d: float, alpha: float) -> float:
        ''' Returns the linearization of state (d, alpha) into a signal 6*d+alpha. '''
        return 6*d+alpha

    def correct(self, y: float) -> np.ndarray:
        for i, (mu, sigma) in enumerate(self.gaussians):
            self.pr[i] = gaussian(mu, sigma).pdf(y)
        self.bel *= self.pr
        self.bel /= np.sum(self.bel)
        return self.bel

    def predict(self, d: float, alpha: float, v: float, w: float, dt: float) -> np.ndarray:
        x = Agent.linearize(d, alpha)
        delta = 6 * (v * math.sin(alpha) * dt) + (w * dt)
        delta = np.clip(delta, -2, 2)
        novoBel = np.zeros_like(self.bel)
        for i, (a, b) in enumerate(self.cells):
            x_prime = (a + b) / 2
            prob = 0.0
            for j, (aPrev, bPrev) in enumerate(self.cells):
                xPrev = (aPrev + bPrev) / 2
                xEsperado = xPrev + delta
                prob += gaussian(xEsperado, self.transition_sigma).pdf(x_prime) * self.bel[j]
            novoBel[i] = prob
        if np.sum(novoBel) == 0:
            novoBel = np.ones_like(novoBel) / len(novoBel)
        else:
            novoBel /= np.sum(novoBel)
        self.bel = novoBel
        return self.bel

    def find_cell(self, y: float):
        ''' Finds which cell y belongs to. '''
        a, b = 0, len(self.cells)
        while a < b:
            k = (a+b)//2
            l, h = self.cells[k]
            if l <= y < h: return k
            if y < l: b = k
            else: a = k+1
        return -1

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline)/self.radius
        return V_l, V_r

    def preprocess(self) -> (float, float, float):
        I = cv2.resize(self.env.front()[180:,:,:], (80, 42))/255
        d, alpha = self.pose_estimator.predict(I.reshape((-1, 42, 80, 3)))[0]
        y = Agent.linearize(d, alpha)
        return y, d, alpha

    def send_commands(self, dt):
        ''' Agent control loop '''
        pwm_left, pwm_right = 0, 0

        y, d, alpha = self.preprocess()
        # Apply correction.
        self.correct(y)

        # Get MAP.
        l, h = self.cells[np.argmax(self.bel)]
        map_y = (l+h)/2

        # Paste your PID controller here.
        error = -map_y
        self.integralError += error * dt
        derivativeError = (error - self.previousError) / dt
        self.previousError = error
        rotation = (
            self.Kp * error +
            self.Ki * self.integralError +
            self.Kd * derivativeError
        )
        velocity = float(np.clip(0.2*(1 - 0.7*min(1.0, abs(error)/0.6)), 0.10, 0.25))

        # Apply prediction.
        self.predict(d, alpha, velocity, rotation, dt)

        pwm_left, pwm_right = self.get_pwm_control(velocity, rotation)
        _, r, _, _ = self.env.step(pwm_left, pwm_right)
        self.score += (r-self.score)/self.env.step_count
        self.env.render(text = f", score: {self.score:.3f}")
        self.hist.render(y)

    @staticmethod
    def load_regression_model(filepath: str) -> tensorflow.keras.Model:
        ''' Loads a Tensorflow model. '''
        model = tensorflow.keras.models.load_model(filepath)
        # If you get an error (most likely due to loading a model saved in a newer version of TensorFlow, try using the line below instead)
        # model = tensorflow.keras.models.load_model(filepath, compile=False)
        model.summary()
        return model

def main():
    print("MAC0318 - Assignment 11")
    env = create_env(
        raw_motor_input = True,
        noisy = True,
        mu_l = 0.007123895,
        mu_r = -0.000523123,
        std_l = 1e-7,
        std_r = 1e-7,
        seed = 101,
        map_name = './maps/loop_empty.yaml',
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        user_tile_start = (0, 0),
        distortion = False,
        top_down = False,
        cam_height = 10,
        # is_external_map = True,
        randomize_maps_on_reset = False,
    )

    env.set_view(FRONT_VIEW_MODE)
    env.reset()
    env.render() # show visualization

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        nonlocal agent
        if symbol == key.ESCAPE: # exit simulation
            env.close()
            sys.exit(0)

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
