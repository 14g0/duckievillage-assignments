# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Iago Cesar Tavares de Souza
# NUSP: 17466770
#
# ---
#
# Assignment 10 - Imitation learning
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/imitation/agent.py
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
from duckievillage import create_env, FRONT_VIEW_MODE
import cv2
import tensorflow

# Uncomment the following line if you get an error with the parallel library
# accusing multiple instantiations
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Basic agent.
class Agent:
    def __init__(self, env, randomize: bool = False):
        self.env = env
        self.radius = 0.0318 # R
        self.baseline = env.unwrapped.wheel_dist/2
        self.motor_gain = 0.68*0.0784739898632288
        self.motor_trim = 0.0007500911693361842

        if randomize: self.randomize()

        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

    def randomize(self):
        self.env.objects = []
        f = [self.env.add_static_duckie, self.env.add_static_big_duckie, self.env.add_cone]
        g = [self.env.add_static_duckiebot, self.env.add_cone, self.env.add_static_duckie]
        k = self.env.road_tile_size/2
        s = (-k/2, k/2)
        r = lambda: random.random()*random.choice(s)
        for i, t in enumerate(self.env.grid):
            if (t["kind"] == "floor") and (random.random() > 0.5):
                x, y = t["coords"]
                random.choice(f)(x*self.env.road_tile_size+k+r(), y*self.env.road_tile_size+k+r())
            elif (t["kind"] in self.env.road_tiles) and (random.random() > 0.67):
                x, y = t["coords"]
                px, py = self.env.tile_position(x, y, centered = True)
                random.choice(g)(px+random.random()*k*0.67, py+random.random()*k*0.67)

        self.env.start_pose = self.env.random_road_pose()

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline)/self.radius
        return V_l, V_r

    def preprocess(self) -> float:
        pass

    def send_commands(self, dt):
        pass

# Agent for data collection.
class DataAgent(Agent):
    def __init__(self, environment):
        ''' Initializes agent '''
        super().__init__(environment, randomize = True)

        self.C = 6

        self.images = []
        self.labels = []

        self.Kp, self.Kd, self.Ki = 4, 0, 0
        self.integralError = 0
        self.previousError = 0
        self.rotation = 0

    def preprocess(self) -> float:
        '''Returns the metric to be used as signal for the PID controller.'''
        d, alpha = self.env.lf_target()
        return self.C*d+alpha

    def send_commands(self, dt):
        ''' Agent control loop '''
        pwm_left, pwm_right = 0, 0
        velocity, rotation = 0.0, 0.0


        if self.key_handler[key.UP]:
            pwm_left, pwm_right = 0.5, 0.5
            velocity, rotation = 0.25, 0.0
        elif self.key_handler[key.LEFT]:
            pwm_left, pwm_right = 0.1, 0.3
            velocity, rotation = 0.15, 1.0
        elif self.key_handler[key.DOWN]:
            pwm_left, pwm_right = -0.5, -0.5
            velocity, rotation = -0.25, 0.0
        elif self.key_handler[key.RIGHT]:
            pwm_left, pwm_right = 0.3, 0.1
            velocity, rotation = 0.15, -1.0

        else:
            y = self.preprocess()
            error = -y

            self.integralError += error * dt
            derivativeError = (error - self.previousError) / dt
            self.previousError = error
            rotation = (
                self.Kp * error +
                self.Ki * self.integralError +
                self.Kd * derivativeError
            )
            velocity = float(np.clip(0.2*(1 - 0.7*min(1.0, abs(error)/0.6)), 0.10, 0.25))

            pwm_left, pwm_right = self.get_pwm_control(velocity, rotation)

        if not (math.isfinite(velocity) and math.isfinite(rotation)):
            velocity, rotation = 0.0, 0.0

        velocity = float(np.clip(velocity, -0.5, 0.5))
        rotation = float(np.clip(rotation, -2.0, 2.0))
        pwm_left = float(np.clip(pwm_left, -1.0, 1.0))
        pwm_right = float(np.clip(pwm_right, -1.0, 1.0))

        self.images.append(cv2.resize(self.env.front(), (80, 60)))
        self.labels.append((velocity, rotation))
    
        self.env.step(pwm_left, pwm_right)
        self.env.render()

# Evaluation agent.
class EvaluationAgent(Agent):
    def __init__(self, environment):
        ''' Initializes agent '''
        super().__init__(environment, randomize = True)
        self.pose_estimator = EvaluationAgent.load_regression_model("./assignments/imitation/e2e_cnn.h5")
        self.score = 0

    @staticmethod
    def load_regression_model(filepath: str) -> tensorflow.keras.Model:
        ''' Loads a Tensorflow model. '''
        model = tensorflow.keras.models.load_model(filepath)
        # If you get an error (most likely due to loading a model saved in a newer version of TensorFlow, try using the line below instead)
        # model = tensorflow.keras.models.load_model(filepath, compile=False)
        model.summary()
        return model

    def preprocess(self) -> (float, float):
        '''Returns the metric to be used as signal for the PID controller.'''
        I = cv2.resize(self.env.front(), (80, 60))/255
        v, w = self.pose_estimator.predict(I.reshape((-1, 60, 80, 3)))[0]
        return v, w

    def send_commands(self, dt):
        ''' Agent control loop '''
        pwm_left, pwm_right = 0, 0

        velocity, rotation = self.preprocess()

        pwm_left, pwm_right = self.get_pwm_control(velocity, rotation)
        _, r, _, _ = self.env.step(pwm_left, pwm_right)
        self.score += (r-self.score)/self.env.step_count
        self.env.render(text = f", score: {self.score:.3f}")

def main():
    print("MAC0318 - Assignment 10")
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
        # video_path = "/tmp/regression-cnn.mp4"
    )

    env.set_view(FRONT_VIEW_MODE)
    env.reset()
    env.render() # show visualization

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        nonlocal agent
        if symbol == key.ESCAPE: # exit simulation
            if isinstance(agent, DataAgent):
                np.save("/tmp/images.npy", agent.images, allow_pickle = True)
                np.save("/tmp/labels.npy", agent.labels, allow_pickle = True)
            env.close()
            sys.exit(0)

        elif (symbol == key.E) or (symbol == key.D):
            # Change to evaluation or data collecting agent.
            pyglet.clock.unschedule(agent.send_commands)
            if (symbol == key.E) and (key.E not in agents): agents[key.E] = EvaluationAgent(env)
            agent = agents[symbol]
            pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)

        elif symbol == key.R:
            agent.randomize()
        elif (symbol == key.P) and isinstance(agent, DataAgent):
            agent.paused = not agent.paused
            print("Data collection is:", "paused" if agent.paused else "unpaused")

        env.render()

    # Instantiate agent
    agents = {key.D: DataAgent(env)}
    initial = key.E if (len(sys.argv) > 1) and (sys.argv[1] == 'eval') else key.D
    if initial == key.E: agents[key.E] = EvaluationAgent(env)
    agent = agents[initial]
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    # When it's done, close environment and exit program
    env.close()

if __name__ == '__main__':
    main()
