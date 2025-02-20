import numpy as np

loop_fps=16 # 16 is main one, but can try to run at 24

cs_img_dimension = (150,280) 
cs_game_res = (1024,768)


# params for discretising mouse
mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
mouse_x_lim = (mouse_x_possibles[0],mouse_x_possibles[-1])
mouse_y_lim = (mouse_y_possibles[0],mouse_y_possibles[-1])

# below options are no longer used, are here due to previous agent iterations
IS_CONTRAST = False # whether to add contrast to image, REDUNDANT
FRAMES_STACK = 3 # how many frames to use as input, REDUNDANT
FRAMES_SKIP = 4 # how many frames to skip in between each of the frames stacked together, REDUNDANT
ACTIONS_PREV = 3 # how many previous actions (and rewards?) to use as aux input, REDUNDANT
AUX_INPUT_ON = False # whether to use aux input at all, REDUNDANT
DATA_STEP = 1 # whether to skip through training data (=1), only use every x steps, REDUNDANT

def mouse_preprocess(mouse_x, mouse_y):
    mouse_x = np.clip(mouse_x, mouse_x_lim[0],mouse_x_lim[1])
    mouse_y = np.clip(mouse_y, mouse_y_lim[0],mouse_y_lim[1])

    mouse_x = min(mouse_x_possibles, key=lambda x_:abs(x_-mouse_x))
    mouse_y = min(mouse_y_possibles, key=lambda x_:abs(x_-mouse_y))

    return mouse_x, mouse_y


def reward_fn(kill, death, shoot):
    return kill - 0.5*death - 0.02*shoot 