
# DON'T FORGET TO RUN "pip install -r requirements.txt" IN THE TERMINAL BEFORE RUNING THE GAME!!!


# Neural Network Data Collection Guide - Air Hockey Game
This document explains the neural network data collection functions in the air hockey game.
Overview
The air hockey game includes two key functions for collecting training data:

get_game_state() - Returns a comprehensive dictionary of all game variables

get_neural_network_input_vector() - Returns a simplified list optimized for neural network input

## Function: get_game_state()
### Description:
Returns a comprehensive dictionary containing all relevant game state information. This function is ideal for analysis, debugging, and understanding the complete game context.

### Return Value Structure:

{
    # Normalized Positions (0.0 to 1.0 range)
    'blue_paddle_x': float,          # Blue paddle X position (0=left edge, 1=right edge)
    'blue_paddle_y': float,          # Blue paddle Y position (0=top edge, 1=bottom edge)
    'red_paddle_x': float,           # Red paddle X position
    'red_paddle_y': float,           # Red paddle Y position
    'disc_x': float,                 # Disc X position
    'disc_y': float,                 # Disc Y position
    'disc_velocity_x': float,        # Disc X velocity (-1.0 to 1.0 range)
    'disc_velocity_y': float,        # Disc Y velocity (-1.0 to 1.0 range)
    
    # Raw Positions (actual pixel coordinates)
    'blue_paddle_x_raw': float,      # Blue paddle X in pixels
    'blue_paddle_y_raw': float,      # Blue paddle Y in pixels
    'red_paddle_x_raw': float,       # Red paddle X in pixels
    'red_paddle_y_raw': float,       # Red paddle Y in pixels
    'disc_x_raw': float,             # Disc X in pixels
    'disc_y_raw': float,             # Disc Y in pixels
    'disc_velocity_x_raw': float,    # Disc X velocity in pixels/frame
    'disc_velocity_y_raw': float,    # Disc Y velocity in pixels/frame
    
    # Goal Area Detection (Boolean flags)
    'blue_paddle_in_own_goal': bool,      # True if blue paddle is in left goal area
    'blue_paddle_in_opponent_goal': bool, # True if blue paddle is in right goal area
    'red_paddle_in_own_goal': bool,       # True if red paddle is in right goal area
    'red_paddle_in_opponent_goal': bool,  # True if red paddle is in left goal area
    
    # Game Status
    'score1': int,                   # Player 1 (blue) score
    'score2': int,                   # Player 2 (red) score
    'serve_direction': int,          # 1 or -1, indicates serve direction
    
    # Calculated Features
    'blue_paddle_to_disc_distance': float,  # Euclidean distance between blue paddle and disc
    'red_paddle_to_disc_distance': float,   # Euclidean distance between red paddle and disc
    'game_time': float                      # Game time in seconds
}

### Usage Example:
# Get complete game state
state = get_game_state()

# Access specific values
blue_x = state['blue_paddle_x']  # Normalized position (0.0-1.0)
disc_speed = math.sqrt(state['disc_velocity_x']**2 + state['disc_velocity_y']**2)
is_defensive_position = state['blue_paddle_in_own_goal']

# Use for analysis or logging
print(f"Blue paddle at ({state['blue_paddle_x_raw']:.1f}, {state['blue_paddle_y_raw']:.1f})")
print(f"Distance to disc: {state['blue_paddle_to_disc_distance']:.1f} pixels")

## Function: get_neural_network_input_vector()

### Description:
Returns a simplified list of 8 essential features optimized for neural network input. This function provides the minimum necessary information while maintaining spatial relationships that the neural network can learn.

### Return Value Structure:

[
    blue_paddle_x,      # Index 0: Blue paddle X position (normalized 0.0-1.0)
    blue_paddle_y,      # Index 1: Blue paddle Y position (normalized 0.0-1.0)
    red_paddle_x,       # Index 2: Red paddle X position (normalized 0.0-1.0)
    red_paddle_y,       # Index 3: Red paddle Y position (normalized 0.0-1.0)
    disc_x,             # Index 4: Disc X position (normalized 0.0-1.0)
    disc_y,             # Index 5: Disc Y position (normalized 0.0-1.0)
    disc_velocity_x,    # Index 6: Disc X velocity (normalized, typically -1.0 to 1.0)
    disc_velocity_y     # Index 7: Disc Y velocity (normalized, typically -1.0 to 1.0)
]

### Usage Example:

# Get neural network input
input_vector = get_neural_network_input_vector()

# Use directly with neural network
import numpy as np
input_array = np.array(input_vector)
prediction = neural_network.predict(input_array.reshape(1, -1))

# Access specific features by index
blue_paddle_x = input_vector[0]
disc_velocity_x = input_vector[6]