import pygame
import time
import math
from pygame.locals import *

pygame.init()

white = (255,255,255)
black = (0,0,0)
green = (0,150,0)
red = (255,0,0)
blue = (0,0,255)
light_blue = (147,251,253)

#Clock initialized
clock= pygame.time.Clock()
#Board Size
screen= pygame.display.set_mode((800,600))
#dividing line
divline1 = screen.get_width()/2, 0
divline2 = screen.get_width()/2 ,screen.get_height()
#Caption
pygame.display.set_caption('Air Hockey!')
#Font Sizes
smallfont = pygame.font.SysFont("comicsansms" , 25)
medfont = pygame.font.SysFont("comicsansms" , 45)
largefont = pygame.font.SysFont("comicsansms" , 65)

# --- GLOBAL GAME OBJECTS (Now using circles for paddles and disc) ---
# Note: Goals remain Rects as they are rectangular areas for scoring.
goalheight = 50
goalwidth = 10 # Adjusted goalwidth to be consistent with original rect width
goal1 = pygame.Rect(0, screen.get_height()/2 - goalheight, goalwidth, goalheight * 2) # Adjusted height to be 100
goal2 = pygame.Rect(screen.get_width() - goalwidth, screen.get_height()/2 - goalheight, goalwidth, goalheight * 2) # Adjusted height to be 100

# Paddle and Disc Properties (Radius and initial center)
# Original paddles were 20x20, so radius is 10 for a similar visual size (diameter 20)
PADDLE_RADIUS = 20
DISC_RADIUS = 15 # Original disc was 20x20, so radius 10. Made slightly larger for better collision visibility.

paddle1 = {
    'center_x': screen.get_width() / 2 - 200,
    'center_y': screen.get_height() / 2,
    'radius': PADDLE_RADIUS
}
paddle2 = {
    'center_x': screen.get_width() / 2 + 200,
    'center_y': screen.get_height() / 2,
    'radius': PADDLE_RADIUS
}
paddleVelocity = 8 # Increased paddle velocity for snappier movement

disc = {
    'center_x': screen.get_width() / 2,
    'center_y': screen.get_height() / 2,
    'radius': DISC_RADIUS
}

# Image loading - you might need to adjust their size if they are not 2x radius already
# For images, we'll draw them centered on the circle's position
img = pygame.image.load('./disc.png')
# Scale disc image to match disc diameter
img = pygame.transform.scale(img, (DISC_RADIUS * 2, DISC_RADIUS * 2))

bluepadimg = pygame.image.load('./bluepad.png')
redpadimg = pygame.image.load('./redpad.png')
# Scale paddle images to match paddle diameter
bluepadimg = pygame.transform.scale(bluepadimg, (PADDLE_RADIUS * 2, PADDLE_RADIUS * 2))
redpadimg = pygame.transform.scale(redpadimg, (PADDLE_RADIUS * 2, PADDLE_RADIUS * 2))


discVelocity = [7, 7] # Increased disc velocity
score1,score2 = 0,0
serveDirection=1

# --- NEURAL NETWORK TRAINING VARIABLES ---
# These variables will be updated every game loop iteration
# and can be used as input features for your neural network

def get_game_state():
    """
    Returns the current game state as a dictionary.
    This function extracts all the relevant variables for neural network training.
    """
    # Check if paddles are in goal areas
    blue_paddle_in_own_goal = is_paddle_in_goal_area(paddle1, 'left')
    #blue_paddle_in_opponent_goal = is_paddle_in_goal_area(paddle1, 'right')
    red_paddle_in_own_goal = is_paddle_in_goal_area(paddle2, 'right')
    #red_paddle_in_opponent_goal = is_paddle_in_goal_area(paddle2, 'left')
    
    game_state = {
        # Paddle positions (normalized to 0-1 range)
        'blue_paddle_x': paddle1['center_x'] / screen.get_width(),
        'blue_paddle_y': paddle1['center_y'] / screen.get_height(),
        'red_paddle_x': paddle2['center_x'] / screen.get_width(),
        'red_paddle_y': paddle2['center_y'] / screen.get_height(),
        
        # Disc position and velocity (normalized)
        'disc_x': disc['center_x'] / screen.get_width(),
        'disc_y': disc['center_y'] / screen.get_height(),
        'disc_velocity_x': discVelocity[0] / 10.0,  # Normalize by max expected velocity
        'disc_velocity_y': discVelocity[1] / 10.0,
        
        # Raw positions (if you prefer absolute coordinates)
        'blue_paddle_x_raw': paddle1['center_x'],
        'blue_paddle_y_raw': paddle1['center_y'],
        'red_paddle_x_raw': paddle2['center_x'],
        'red_paddle_y_raw': paddle2['center_y'],
        'disc_x_raw': disc['center_x'],
        'disc_y_raw': disc['center_y'],
        'disc_velocity_x_raw': discVelocity[0],
        'disc_velocity_y_raw': discVelocity[1],
        
        # Goal area detection
        'blue_paddle_in_own_goal': blue_paddle_in_own_goal,
        #'blue_paddle_in_opponent_goal': blue_paddle_in_opponent_goal,
        'red_paddle_in_own_goal': red_paddle_in_own_goal,
        #'red_paddle_in_opponent_goal': red_paddle_in_opponent_goal,
        
        # Additional useful features
        'score1': score1,
        'score2': score2,
        'serve_direction': serveDirection,
        
        # Calculated distances (could be useful features)
        'blue_paddle_to_disc_distance': get_distance(
            (paddle1['center_x'], paddle1['center_y']),
            (disc['center_x'], disc['center_y'])
        ),
        'red_paddle_to_disc_distance': get_distance(
            (paddle2['center_x'], paddle2['center_y']),
            (disc['center_x'], disc['center_y'])
        ),
        
        # Time step (if you want to track game progression)
        'game_time': pygame.time.get_ticks() / 1000.0  # Convert to seconds
    }
    
    return game_state

def get_neural_network_input_vector():
    """
    Returns a list/array of values that can be directly fed into a neural network.
    This is a flattened version of the game state with only the essential features.
    """
    state = get_game_state()
    
    # Essential features only - let the neural network learn spatial relationships
    input_vector = [
        state['blue_paddle_x'],      # Blue paddle X position (normalized 0-1)
        state['blue_paddle_y'],      # Blue paddle Y position (normalized 0-1)
        state['red_paddle_x'],       # Red paddle X position (normalized 0-1)
        state['red_paddle_y'],       # Red paddle Y position (normalized 0-1)
        state['disc_x'],             # Disc X position (normalized 0-1)
        state['disc_y'],             # Disc Y position (normalized 0-1)
        state['disc_velocity_x'],    # Disc X velocity (normalized)
        state['disc_velocity_y'],    # Disc Y velocity (normalized)
    ]
    
    return input_vector

# --- Helper Functions for Circle Collision and Positioning ---

def get_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_paddle_in_goal_area(paddle, goal_side):
    """
    Checks if a paddle is inside a goal area semicircle.
    
    Args:
        paddle: paddle dictionary with center_x, center_y, radius
        goal_side: 'left' for blue goal area, 'right' for red goal area
    
    Returns:
        Boolean: True if paddle is inside the goal area semicircle
    """
    goal_circle_radius = screen.get_width() / 10
    paddle_center = (paddle['center_x'], paddle['center_y'])
    
    if goal_side == 'left':
        # Blue goal area - semicircle centered at (0, screen_height/2)
        goal_center = (0, screen.get_height() / 2)
        
        # Check if paddle is within the semicircle radius
        distance_to_goal_center = get_distance(paddle_center, goal_center)
        if distance_to_goal_center <= goal_circle_radius:
            # Additional check: must be on the right side of the goal line (x >= 0)
            # and within the semicircle arc (right half of the circle)
            return paddle['center_x'] >= 0
    
    elif goal_side == 'right':
        # Red goal area - semicircle centered at (screen_width, screen_height/2)
        goal_center = (screen.get_width(), screen.get_height() / 2)
        
        # Check if paddle is within the semicircle radius
        distance_to_goal_center = get_distance(paddle_center, goal_center)
        if distance_to_goal_center <= goal_circle_radius:
            # Additional check: must be on the left side of the goal line (x <= screen_width)
            # and within the semicircle arc (left half of the circle)
            return paddle['center_x'] <= screen.get_width()
    
    return False

def check_circle_collision(circle1, circle2):
    """Checks for collision between two circles."""
    distance = get_distance(
        (circle1['center_x'], circle1['center_y']),
        (circle2['center_x'], circle2['center_y'])
    )
    return distance <= (circle1['radius'] + circle2['radius'])

def get_rect_from_circle(circle):
    """Creates a pygame.Rect that encloses a circle. Useful for blitting images."""
    return pygame.Rect(
        circle['center_x'] - circle['radius'],
        circle['center_y'] - circle['radius'],
        circle['radius'] * 2,
        circle['radius'] * 2
    )

def resetPuck():
    global serveDirection # Use global to modify the variable in the outer scope
    discVelocity[0] = 7 * serveDirection # Reset velocity with serve direction
    discVelocity[1] = 7 * serveDirection # Reset velocity with serve direction
    print(f"Score: Player 1: {score1}, Player 2: {score2}")
    disc['center_x'] = screen.get_width() / 2
    disc['center_y'] = screen.get_height() / 2
    # Ensure disc is not stuck in a paddle after reset
    # This might need more sophisticated handling in a real game
    time.sleep(0.5) # Small delay to prevent immediate re-collision

def text_objects(text,color,size):
    if size == "small":
        textSurface = smallfont.render(text , True , color)
    elif size == "medium":
        textSurface = medfont.render(text , True , color)
    elif size == "large":
        textSurface = largefont.render(text , True , color)
    return textSurface , textSurface.get_rect()

def pause():
    paused = True
    message_to_screen("Paused",black,-100,size="large")
    message_to_screen("Press c to continue , q to quit",black,25)
    pygame.display.update()
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    paused = False

                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()
        clock.tick(5)

def message_to_screen(msg,color,y_displace=0,x_displace=0,size = "small"):
    textSurf, textRect = text_objects(msg,color,size)
    textRect.center = (screen.get_width()/2+x_displace) , ((screen.get_height()/2) + y_displace)
    screen.blit(textSurf,textRect)


def gameLoop():
    global score1, score2, serveDirection # Declare globals to modify them
    gameExit = False
    gameOver = False
    score1, score2 = 0, 0
    
    # --- DATA COLLECTION FOR NEURAL NETWORK ---
    # Lists to store game states and actions for training
    game_states = []
    player1_actions = []
    player2_actions = []

    while not gameExit:

        for event in pygame.event.get():
            # Reset movement flags each frame for smooth input handling
            # Better way: get_pressed() outside the event loop
            if event.type == pygame.QUIT:
                gameExit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause()

        # --- COLLECT CURRENT GAME STATE FOR NEURAL NETWORK ---
        current_state = get_game_state()
        current_input_vector = get_neural_network_input_vector()
        
        # Example: Print the game state every 60 frames (once per second at 60 FPS)
        if pygame.time.get_ticks() % 1000 < 17:  # Approximately every second
            print("Current Game State:")
            print(f"Blue Paddle: ({current_state['blue_paddle_x_raw']:.1f}, {current_state['blue_paddle_y_raw']:.1f})")
            print(f"Red Paddle: ({current_state['red_paddle_x_raw']:.1f}, {current_state['red_paddle_y_raw']:.1f})")
            print(f"Disc: ({current_state['disc_x_raw']:.1f}, {current_state['disc_y_raw']:.1f})")
            print(f"Disc Velocity: ({current_state['disc_velocity_x_raw']:.1f}, {current_state['disc_velocity_y_raw']:.1f})")
            
            # Print goal area status
            print("Goal Area Status:")
            print(f"  Blue paddle in own goal: {current_state['blue_paddle_in_own_goal']}")
            #print(f"  Blue paddle in opponent goal: {current_state['blue_paddle_in_opponent_goal']}")
            print(f"  Red paddle in own goal: {current_state['red_paddle_in_own_goal']}")
            #print(f"  Red paddle in opponent goal: {current_state['red_paddle_in_opponent_goal']}")
            
            print(f"Neural Network Input Vector: {[f'{x:.3f}' for x in current_input_vector]}")
            print("-" * 50)

        # Input handling using pygame.key.get_pressed() for continuous movement
        keys = pygame.key.get_pressed()
        
        # --- RECORD PLAYER ACTIONS FOR TRAINING ---
        player1_action = [0, 0, 0, 0]  # [left, right, up, down]
        player2_action = [0, 0, 0, 0]  # [left, right, up, down]
        
        # Player 1 (Blue Paddle - WASD)
        if keys[K_a]:
            paddle1['center_x'] -= paddleVelocity
            player1_action[0] = 1
        if keys[K_d]:
            paddle1['center_x'] += paddleVelocity
            player1_action[1] = 1
        if keys[K_w]:
            paddle1['center_y'] -= paddleVelocity
            player1_action[2] = 1
        if keys[K_s]:
            paddle1['center_y'] += paddleVelocity
            player1_action[3] = 1

        # Player 2 (Red Paddle - Arrow Keys)
        if keys[K_LEFT]:
            paddle2['center_x'] -= paddleVelocity
            player2_action[0] = 1
        if keys[K_RIGHT]:
            paddle2['center_x'] += paddleVelocity
            player2_action[1] = 1
        if keys[K_UP]:
            paddle2['center_y'] -= paddleVelocity
            player2_action[2] = 1
        if keys[K_DOWN]:
            paddle2['center_y'] += paddleVelocity
            player2_action[3] = 1

        # Store the data for training (you can save this to a file later)
        game_states.append(current_input_vector)
        player1_actions.append(player1_action)
        player2_actions.append(player2_action)

        # --- Update Paddle Positions and Boundaries (Circle-based) ---

        # Paddle 1 Boundaries
        if paddle1['center_y'] - paddle1['radius'] < 0:
            paddle1['center_y'] = paddle1['radius']
        elif paddle1['center_y'] + paddle1['radius'] > screen.get_height():
            paddle1['center_y'] = screen.get_height() - paddle1['radius']
        if paddle1['center_x'] - paddle1['radius'] < 0:
            paddle1['center_x'] = paddle1['radius']
        # Left half of the screen for paddle1
        elif paddle1['center_x'] + paddle1['radius'] > screen.get_width() / 2:
            paddle1['center_x'] = screen.get_width() / 2 - paddle1['radius']

        # Paddle 2 Boundaries
        if paddle2['center_y'] - paddle2['radius'] < 0:
            paddle2['center_y'] = paddle2['radius']
        elif paddle2['center_y'] + paddle2['radius'] > screen.get_height():
            paddle2['center_y'] = screen.get_height() - paddle2['radius']
        # Right half of the screen for paddle2
        if paddle2['center_x'] + paddle2['radius'] > screen.get_width():
            paddle2['center_x'] = screen.get_width() - paddle2['radius']
        elif paddle2['center_x'] - paddle2['radius'] < screen.get_width() / 2:
            paddle2['center_x'] = screen.get_width() / 2 + paddle2['radius']

        # --- Update Disc Position ---
        disc['center_x'] += discVelocity[0]
        disc['center_y'] += discVelocity[1]

        # --- Disc Wall Collisions ---
        # Top and Bottom walls
        if disc['center_y'] - disc['radius'] < 0 or disc['center_y'] + disc['radius'] > screen.get_height():
            discVelocity[1] *= -1
            # Prevent sticking to walls by slightly moving the disc back
            if disc['center_y'] - disc['radius'] < 0:
                disc['center_y'] = disc['radius']
            else:
                disc['center_y'] = screen.get_height() - disc['radius']

        # Left and Right walls (excluding goal areas)
        # Left wall
        if disc['center_x'] - disc['radius'] < 0:
            # Check if it's within goal height
            if disc['center_y'] > goal1.top and disc['center_y'] < goal1.bottom:
                score2 += 1
                serveDirection = -1
                resetPuck()
            else:
                discVelocity[0] *= -1
                disc['center_x'] = disc['radius'] # Move out of wall

        # Right wall
        if disc['center_x'] + disc['radius'] > screen.get_width():
            # Check if it's within goal height
            if disc['center_y'] > goal2.top and disc['center_y'] < goal2.bottom:
                score1 += 1
                serveDirection = 1
                resetPuck()
            else:
                discVelocity[0] *= -1
                disc['center_x'] = screen.get_width() - disc['radius'] # Move out of wall

        # --- Circle-to-Circle Collisions (Paddles and Disc) ---
        # A more robust collision would calculate bounce angle based on impact point.
        # For simplicity, we'll just reverse X velocity if collision occurs.
        # This will make it act more like a pong paddle.

        if check_circle_collision(disc, paddle1):
            # Reverse X velocity
            discVelocity[0] *= -1
            # Adjust position to prevent sticking
            # Move disc out of the paddle
            overlap = (disc['radius'] + paddle1['radius']) - get_distance((disc['center_x'], disc['center_y']), (paddle1['center_x'], paddle1['center_y']))
            if overlap > 0:
                # Calculate vector from paddle to disc
                dx = disc['center_x'] - paddle1['center_x']
                dy = disc['center_y'] - paddle1['center_y']
                # Normalize the vector
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx /= length
                    dy /= length
                    disc['center_x'] += dx * overlap
                    disc['center_y'] += dy * overlap
                else: # Centers are identical, just push along X
                    disc['center_x'] += 1 if discVelocity[0] > 0 else -1

        if check_circle_collision(disc, paddle2):
            # Reverse X velocity
            discVelocity[0] *= -1
            # Adjust position to prevent sticking
            overlap = (disc['radius'] + paddle2['radius']) - get_distance((disc['center_x'], disc['center_y']), (paddle2['center_x'], paddle2['center_y']))
            if overlap > 0:
                dx = disc['center_x'] - paddle2['center_x']
                dy = disc['center_y'] - paddle2['center_y']
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx /= length
                    dy /= length
                    disc['center_x'] += dx * overlap
                    disc['center_y'] += dy * overlap
                else: # Centers are identical, just push along X
                    disc['center_x'] += 1 if discVelocity[0] > 0 else -1


        # --- Render Logic ---
        screen.fill(black)
        message_to_screen("Player 1",white,-250,-150,"small")
        message_to_screen(str(score1),white,-200,-150,"small")
        message_to_screen("Player 2",white,-250,150,"small")
        message_to_screen(str(score2),white,-200,150,"small")

        # Draw paddles as circles
        # pygame.draw.circle(surface, color, center_position, radius, width=0)
        pygame.draw.circle(screen, (255,100, 100), (int(paddle1['center_x']), int(paddle1['center_y'])), paddle1['radius'])
        pygame.draw.circle(screen, (20,20,100), (int(paddle2['center_x']), int(paddle2['center_y'])), paddle2['radius'])

        # Draw goals (still rects)
        pygame.draw.rect(screen,light_blue,goal1)
        pygame.draw.rect(screen,light_blue,goal2)

        # Blit images for disc and paddles, centering them on the circle's position
        disc_rect_for_blit = get_rect_from_circle(disc)
        screen.blit(img, disc_rect_for_blit.topleft)

        bluepad_rect_for_blit = get_rect_from_circle(paddle1)
        redpad_rect_for_blit = get_rect_from_circle(paddle2)
        screen.blit(bluepadimg, bluepad_rect_for_blit.topleft)
        screen.blit(redpadimg, redpad_rect_for_blit.topleft)


        # Drawing game boundaries and center line
        center_circle_radius = screen.get_width()/10
        pygame.draw.circle(screen, white , (screen.get_width()/2, screen.get_height()/2), center_circle_radius, 5)
        pygame.draw.line(screen , white , divline1, divline2 ,5 )
        
        # Draw semicircles for goal areas (same radius as center circle)
        goal_circle_radius = center_circle_radius
        
        # Left goal semicircle (blue side) - facing right
        left_goal_rect = pygame.Rect(
            -goal_circle_radius,  # Start at negative radius so semicircle touches the goal line
            screen.get_height()/2 - goal_circle_radius,
            goal_circle_radius * 2,
            goal_circle_radius * 2
        )
        pygame.draw.arc(screen, blue, left_goal_rect, -math.pi/2, math.pi/2, 5)  # Right semicircle
        
        # Right goal semicircle (red side) - facing left  
        right_goal_rect = pygame.Rect(
            screen.get_width() - goal_circle_radius,  # Position so semicircle touches the goal line
            screen.get_height()/2 - goal_circle_radius,
            goal_circle_radius * 2,
            goal_circle_radius * 2
        )
        pygame.draw.arc(screen, red, right_goal_rect, math.pi/2, 3*math.pi/2, 5)  # Left semicircle
        
        # Original boundary lines
        pygame.draw.line(screen, blue,(0,0), (screen.get_width()/2 - 5,0) ,5)
        pygame.draw.line(screen, blue,(0,screen.get_height()), (screen.get_width()/2 - 5,screen.get_height()) ,5)
        pygame.draw.line(screen, red, (screen.get_width()/2+5,0), (screen.get_width() ,0) ,5)
        pygame.draw.line(screen, red, (screen.get_width()/2 + 5,screen.get_height()) , (screen.get_width(),screen.get_height()) ,5)
        pygame.draw.line(screen, blue, (0,0), (0,screen.get_height()/2-goalheight) ,5)
        pygame.draw.line(screen, blue, (0,screen.get_height()/2 + goalheight), (0,screen.get_height()) ,5)
        pygame.draw.line(screen, red, (screen.get_width(),0), (screen.get_width(),screen.get_height()/2-goalheight) ,5)
        pygame.draw.line(screen, red, (screen.get_width(),screen.get_height()/2 + goalheight), (screen.get_width(),screen.get_height()) ,5)

        pygame.display.update()
        clock.tick(60) # Increased tick rate for smoother circle movement
    
    # --- IF WE WANT TO SAVE TRAINING DATA WHEN GAME ENDS ---
    print(f"Game ended. Collected {len(game_states)} training samples.")
    # You can save this data to files for later neural network training
    # Example: 
    # import pickle
    # with open('training_data.pkl', 'wb') as f:
    #     pickle.dump({'states': game_states, 'player1_actions': player1_actions, 'player2_actions': player2_actions}, f)

gameLoop()
pygame.quit() # Ensure pygame is properly quit when the game loop ends
quit() # Exit the Python script