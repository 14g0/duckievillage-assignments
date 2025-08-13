# MAC0318 Programação de Robôs Móveis
#
# Preencha o cabeçalho abaixo com seus dados. 
# Se o trabalho for feito em grupo, coloque os nomes de 
# todos os integrantes (copie e cole as linhas abaixo)
#
# Nome: Iago Cesar Tavares de Souza
# NUSP: 17466770
#
# ---
#
# Atividade - Direção manual
#
# Objetivo:
#  - Escrever um agente controlado pelo teclado no simulador.
#
# Não se esqueça de executar este arquivo do diretório raiz do Duckievillage (p.ex.):
#   cd ~/duckievillage
#   source bin/activate 
#   python3 assignments/manual/manual.py
#
# Instruções:
#  0. Preencha o cabeçalho acima
#  1. Assegure-se que o código submetido é o correto e funcionando. Se seu código não roda, você receberá nota 0 na atividade.
#  2. Submeta este arquivo no e-disciplinas.

import sys
import pyglet
from pyglet.window import key
from duckievillage import create_env

class Agent:
    # Agent initialization
    def __init__(self, environment):
        """ Initializes agent """
        # KeyStateHandler handles key states.
        key_handler = key.KeyStateHandler()
        # Let's register our key handler to the environment's key listener.
        environment.unwrapped.window.push_handlers(key_handler)
        self.env = environment
        self.key_handler = key_handler

    def send_commands(self, dt):
        ''' Agent control loop '''
        # This function updates the robot behaviour. Parameter dt is the elapsed time, in
        # milliseconds, since the last update call.
        # At each step, the agent produces an action in the form of two reals in [-1,1]:
        #   pwm_left, pwm_right = left motor power, right motor power
        # Play with these values and figure out how to make your own remote control duckiebot!
        pwm_left, pwm_right = 0, 0

        # The key_handler object handles keyboard events. It's basically a map indexed by Pyglet
        # keys with values True if the key is being held, or False otherwise.
        if self.key_handler[key.UP]:
            # Set the values of pwm_left and pwm_right to make your robot move forward
            # at a reasonable speed (perhaps add acceleration/deacceleration)
            print('Move Forward!')
            pwm_left, pwm_right = 0.8, 0.8

        if self.key_handler[key.LEFT]:
            # Set the values of pwm_left and pwm_right to make your robot turn left
            print('Turn Left!')
            pwm_left, pwm_right = 0.5, 0.8

        if self.key_handler[key.DOWN]:
            # Set the values of pwm_left and pwm_right to make your robot move backward
            print('Move Backward!')
            pwm_left, pwm_right = -0.8, -0.8

        if self.key_handler[key.RIGHT]:
            # Set the values of pwm_left and pwm_right to make your robot turn right
            print('Turn Right!')
            pwm_left, pwm_right = 0.8, 0.5

        # At each step, the environment may (or may not) change given your actions. Function step takes
        # as parameter the two motor powers as action and returns an observation (what the robot is
        # currently seeing), a reward (mostly used for reinforcement learning), whether the episode is
        # done (also used for reinforcement learning) and some info on the elapsed episode.  Let's ignore
        # return values for now.
        obs, reward, done, info = self.env.step(pwm_left, pwm_right)

        # Refresh at every update.
        self.env.render()

def main():
    print("MAC0318 - Assignment 1")
    # We'll use our version of Duckietown: Duckievillage. This environment will be where we'll run most
    # our tasks.
    env = create_env(
        raw_motor_input = True,
        seed = 101,
        map_name = 'loop_empty',
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        distortion = False,
        top_down = False
    )

    # Let's reset the environment to get our Duckiebot somewhere random.
    env.reset()
    # This function is used to draw the environment to a graphical user interface using Pyglet.
    env.render()

    # We use this function for on-press key events (not something we use for real-time feedback,
    # though). We'll register ESC as our way out of the Matrix.
    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE:
            env.close()
            sys.exit(0)
        env.render()

    # Instantiante agent
    agent = Agent(env)
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    env.close()

if __name__ == '__main__':
    main()
