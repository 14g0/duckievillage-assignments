###
### TODO: Replace Tensorflow code with PyTorch
###
# MAC0318 - Intro to Robotics
#
# Name:
# NUSP:    
#
# ---
#
# Assignment 8 - Lane following as a regression task
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/duckievillage
#   git pull
#   source bin/activate
#   python3 assignments/learning/agent.py
#
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change hasn't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file via e-disciplinas.

import sys
import pyglet
import numpy as np
import math
from pyglet.window import key
from duckievillage import create_env, FRONT_VIEW_MODE
import cv2
#import tensorflow

# Uncomment the following line if you get an error with the parallel library
# accusing multiple instantiations
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Agent:
    # Agent initialization
    def __init__(self, environment):
        ''' Initializes agent '''
        self.env = environment
        # Wheel radius
        self.radius = 0.0318 # R
        # Distance between wheels
        self.baseline = environment.unwrapped.wheel_dist # 2L = 0.102 [m]
        # Motor constants
        self.motor_gain = 0.68*0.0784739898632288 # K_m
        self.motor_trim = 0.0007500911693361842 # K_t
        # Filter
        # Yellow colow segmentation filter
        self.yellow_lower = np.array([10, 80, 100], dtype=np.uint8)  
        self.yellow_upper = np.array([30, 255, 255], dtype=np.uint8) 
        # White color segmentation filter
        self.white_lower = np.array([0, 0, 180], dtype=np.uint8)     
        self.white_upper = np.array([179, 80, 255], dtype=np.uint8)   
        # horizon threshold -- to crop image below horizon 
        self.horizon = 180 # in pixels, considering a 800x600 image size
        ######################################################################
        # Regressor - Replace with your model's filepath
        #self.pose_estimator = Agent.load_regression_model("assignments/regression/mlp_lane_pose_estimation.h5")
        ######################################################################
        # Controller
        key_handler = key.KeyStateHandler()
        environment.unwrapped.window.push_handlers(key_handler)
        self.velocity = 0.2 # robot's logitudinal velocity
        self.rotation = 0.0 # robot's angular velocity
        self.key_handler = key_handler

    # @staticmethod
    # def load_regression_model(filepath: str) -> tensorflow.keras.Model:
    #     ''' Loads a Tensorflow model. '''
    #     #model = keras.models.load_model(filepath)
    #     # If you get an error (most likely due to loading a model saved in a newer version of TensorFlow, try using the line below instead)
    #     model = tensorflow.keras.models.load_model(filepath, compile=False)
    #     print(model.summary())
    #     return model

    def preprocess(self) -> float:
        ''' Returns the metric to be used as signal for the PID controller. '''
        I = self.env.front() # capture image from the robot's front camera        
        
        I = cv2.resize(I[self.horizon:,:,:], (80, 42)) # crop and resize it to reduce processing cost
        # segment colors
        hsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)
        img = np.zeros((I.shape[0],I.shape[1],2), dtype=np.float32) # create 2-channel image
        img[:,:,0] = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)//255 # yellow dashed line
        img[:,:,1] = cv2.inRange(hsv, self.white_lower, self.white_upper)//255 # white line

        if self.key_handler[key.K]: # Capture image -- good for debuging and further improving model
            import matplotlib.pyplot as plt
            image = np.zeros((42,80,3), dtype=np.float32) 
            image[:,:,:2] = img[:,:,:]
            plt.imshow(image)
            filename=f'image-capture-{self.env.unwrapped.step_count}.png'
            plt.savefig(filename)
            print("  Saved image to ", filename)
            
        # now run your regressor to predict the relative pose value
        #  the first dimension of the input is the batch size: we are using a batch of 1 instance, 
        #  so we need to reshape the input to match the expected form -- also, tensorflow outputs tensors, 
        #   and we expect a real value, so we just get the single element of the output
        #estimate = self.pose_estimator.predict(img.reshape((-1,42,80,2)))[0,0] # note, this should match the resized image size above
        estimate = 0.0 
        return estimate # should approximate value of y = 6*d + alpha

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline/2)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline/2)/self.radius
        return V_l, V_r

    def send_commands(self, dt) -> None:
        ''' Agent control loop '''
        # Manual control for testing in order to understand the environment.
        # -- You should delete this snippet after your controller is set.
        if self.key_handler[key.W]:
            self.velocity = 0.2
        if self.key_handler[key.A]:
            self.rotation += 0.5
        if self.key_handler[key.S]:
            self.velocity = 0.0
        if self.key_handler[key.D]:
            self.rotation = -0.5
        # -- End of remote control snippet.

        # Target value for lane-following.
        y = self.preprocess()
        # print(y) # uncomment this for debugging

        # Control - replace with your PID controller you developed         
        self.rotation = -5*y
        # Transform into motor duty cicle equivalents
        pwm_left, pwm_right = self.get_pwm_control(self.velocity, self.rotation)
        # Now send commands
        self.env.step(pwm_left, pwm_right) # send commands to motor
        self.env.render() # simulate environment


def main():
    print("MAC0318 - Assignment 8")
    env = create_env(
        raw_motor_input = True,
        noisy = True,
        mu_l = 0.007123895,
        mu_r = -0.000523123,
        std_l = 1e-7,
        std_r = 1e-7,
        seed = 101,
        #map_name = './maps/minimal_udem1.yaml', # use this map to observe the effect of unexpected input (in this case, unseen objects in the images)
        map_name = './maps/loop_empty',
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
    env.render('human')

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE: # exit simulation
            env.close()
            sys.exit(0)
        elif symbol == key.RETURN: # Reset pose.
            env.reset_pos()

        env.render() # show image to user

    agent = Agent(env)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    pyglet.app.run()
    env.close()

if __name__ == '__main__':
    main()

