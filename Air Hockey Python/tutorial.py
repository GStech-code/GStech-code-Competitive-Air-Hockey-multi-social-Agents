from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT
from air_hockey_python import Game
import multiprocessing as mp
from tqdm import tqdm
import pickle
import pygame
import random
import neat
import math
import time
import os


class HockeyGame:
    def __init__(self):
        self.game = Game(render= True)
        self.game.max_score = 2
        self.paddle1 = self.game.paddle1
        self.paddle2 = self.game.paddle2
        self.disc = self.game.disc
        self.screen_width = self.game.screen_width
        self.screen_height = self.game.screen_height



    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Reset the game state before testing
        self.game.reset_game()
        
        # Set paddle control modes
        self.paddle1.player_controlled = False  # AI controlled
        self.paddle2.player_controlled = True   # Human controlled
        
        run = True
        clock = pygame.time.Clock()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        print("Testing AI vs Human Player")
        print("Use arrow keys to control the red paddle")
        print("Press ESC to quit")
        
        while run:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        run = False
                        break
            
            if not run:
                break
            
            # AI controls blue paddle (left side)
            # Get current game state for AI input
            ai_input = self.get_normalized_inputs_for_paddle('left')
            ai_output = net.activate(ai_input)
            ai_decision = self.process_neat_output(ai_output, 'left')
            
            # DEBUG: Print every 60 frames (once per second)
            if pygame.time.get_ticks() % 1000 < 16:  # Roughly once per second
                print(f"AI Input: {[f'{x:.2f}' for x in ai_input]}")
                print(f"AI Output: {[f'{x:.2f}' for x in ai_output]}")
                print(f"AI Decision: {ai_decision}")
                print(f"Any movement: {any(ai_decision.values())}")
                print("---")
            
            if all(value <= 0.5 for value in ai_decision.values()):
                pass
            else:
                self.paddle1.update(ai_decision, self.screen_width, self.screen_height)
            
            # Human controls red paddle (right side)
            keys = pygame.key.get_pressed()
            self.paddle2.update(keys, self.screen_width, self.screen_height)
            
            # Update disc physics
            self.disc.update(self.screen_width, self.screen_height)
            self.disc.check_wall_collision(self.screen_width, self.screen_height)
            
            # Check scoring
            score_left, score_right = self.disc.check_side_collision(
                self.screen_width, self.screen_height, self.game.goal1, self.game.goal2
            )
            if score_left:
                self.game.score1 += 1
                self.game.serve_direction = 1
                self.game.reset_puck()
            elif score_right:
                self.game.score2 += 1
                self.game.serve_direction = -1
                self.game.reset_puck()
            
            # Check paddle collisions
            if self.disc.check_paddle_collision(self.paddle1):
                self.disc.handle_paddle_collision(self.paddle1)
            if self.disc.check_paddle_collision(self.paddle2):
                self.disc.handle_paddle_collision(self.paddle2)
            
            # Clear the screen before drawing
            glClear(GL_COLOR_BUFFER_BIT)
            
            # Render the game
            self.game.draw_field()
            self.game.draw_ui()
            
            # Draw game objects
            self.game.draw_circle(self.disc.center_x, self.disc.center_y, self.disc.radius, self.game.white)
            self.game.draw_circle(self.paddle1.center_x, self.paddle1.center_y, self.paddle1.radius, self.game.blue)
            self.game.draw_circle(self.paddle2.center_x, self.paddle2.center_y, self.paddle2.radius, self.game.red)
            
            # Swap OpenGL buffers (required for OpenGL rendering)
            pygame.display.flip()
            
            # Check win conditions
            if self.game.score1 >= 2 or self.game.score2 >= 2 or (pygame.time.get_ticks() / 1000.0) > 120:
                if (pygame.time.get_ticks() / 1000.0) > 120:
                    print("Time out!")
                else:
                    print(f"Final Score - AI: {self.game.score1}, Human: {self.game.score2}")
                    if self.game.score1 > self.game.score2:
                        print("AI Wins!")
                    else:
                        print("Human Wins!")
                run = False
            
            # Control frame rate
            clock.tick(60)


    def train_ai(self, genome1, genome2, config, render = False):
        render_debug = render  # Set to False for full-speed NEAT training
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        run = True
        max_points = self.game.max_score
        while run:
            # Blue paddle (left)
            blue_inputs = self.get_normalized_inputs_for_paddle('left')
            blue_output = net1.activate(blue_inputs)
            blue_decision = self.process_neat_output(blue_output, 'left')
            #output1 = net1.activate((self.paddle1.center_x / (self.game.screen_width / 2) * -1, self.paddle1.center_y / self.game.screen_height, self.paddle1.current_speed, self.paddle2.center_x - (self.game.screen_width / 2), self.paddle2.center_y, self.disc.center_x - (self.game.screen_width / 2), self.paddle2.current_speed, self.disc.center_y, self.disc.x_vel, self.disc.y_vel)) # Blue
            #decision1 = self.game.neat_requests(output1, 'left')
            if all(value <= 0.5 for value in blue_decision.values()):
                pass
            else:
                self.paddle1.update(blue_decision, self.screen_width, self.screen_height)
            
            red_inputs = self.get_normalized_inputs_for_paddle('right')
            red_output = net2.activate(red_inputs)
            red_decision = self.process_neat_output(red_output, 'right')
            #output2 = net2.activate((self.paddle2.center_x - (self.game.screen_width / 2), self.paddle2.center_y, self.paddle2.current_speed, self.paddle1.center_x / (self.game.screen_width / 2) * -1,  self.paddle1.center_y / self.game.screen_height, self.paddle1.current_speed, self.disc.center_x - (self.game.screen_width / 2), self.disc.center_y, self.disc.x_vel, self.disc.y_vel)) # Red
            #decision2 = self.game.neat_requests(output2, 'right')
            if all(value <= 0.5 for value in red_decision.values()):
                pass
            else:
                self.paddle2.update(red_decision, self.screen_width, self.screen_height)
            
            #print(f"Blue decision: {decision1}\nRed decision: {decision2}")

            # Just advance one frame of game logic, no OpenGL rendering
            game_info = self.game.update_one_frame(blue_decision, red_decision, render= render_debug)
            game_info['blue_paddle_num_in_goal'] += 1 if  game_info['blue_paddle_in_own_goal'] else 0
            game_info['red_paddle_num_in_goal'] += 1 if  game_info['red_paddle_in_own_goal'] else 0


            #if pygame.time.get_ticks() % 1000 < 16:  # Roughly once per second
            #    print(f"Blue output: {[f'{x:.2f}' for x in blue_output]} Any movement of blue: {any(blue_decision.values())}")
            #    print(f"Red output: {[f'{x:.2f}' for x in red_output]} Any movement of red: {any(red_decision.values())}")
            #    print("---")
            self.calculate_fitness(genome1= genome1, genome2= genome2, game_info= game_info)

            if (game_info['score1'] >= max_points) or (self.game.score2 >= max_points) or game_info['game_time'] > 20:
                #if game_info['game_time'] > 20:
                #    print("\nTime out!")
                #else:
                    #print(f"score1 : score2\n     {game_info['score1']} : {game_info['score2']}")
                
                #self.calculate_fitness(genome1= genome1, genome2= genome2, game_info= game_info)
                break
        
        pygame.quit()

    # input normalization and neural network processing
    #def get_normalized_inputs_for_paddle(self, paddle_side):
    #    """Get properly normalized inputs for a paddle"""
    #    if paddle_side == 'left':
    #        # Blue paddle (left side)
    #        paddle = self.paddle1
    #        opponent = self.paddle2
    #        # Normalize paddle position to [-1, 1] for x, [0, 1] for y
    #        paddle_x = (paddle.center_x - self.screen_width/4) / (self.screen_width/4)
    #        paddle_y = paddle.center_y / self.screen_height
    #        # Opponent position relative to paddle
    #        opponent_x = (opponent.center_x - paddle.center_x) / (self.screen_width/2)
    #        opponent_y = (opponent.center_y - paddle.center_y) / self.screen_height
    #        # Disc position relative to paddle
    #        disc_x = (self.disc.center_x - paddle.center_x) / (self.screen_width/2)
    #        disc_y = (self.disc.center_y - paddle.center_y) / self.screen_height
    #    else:
    #        # Red paddle (right side)
    #        paddle = self.paddle2
    #        opponent = self.paddle1
    #        # Normalize paddle position to [-1, 1] for x, [0, 1] for y
    #        paddle_x = (paddle.center_x - 3*self.screen_width/4) / (self.screen_width/4)
    #        paddle_y = paddle.center_y / self.screen_height
    #        # Opponent position relative to paddle
    #        opponent_x = (opponent.center_x - paddle.center_x) / (self.screen_width/2)
    #        opponent_y = (opponent.center_y - paddle.center_y) / self.screen_height
    #        # Disc position relative to paddle
    #        disc_x = (self.disc.center_x - paddle.center_x) / (self.screen_width/2)
    #        disc_y = (self.disc.center_y - paddle.center_y) / self.screen_height
    #    
    #    return [
    #        paddle_x,                           # Paddle X position (normalized)
    #        paddle_y,                           # Paddle Y position (normalized)
    #        paddle.actual_speed / paddle.max_velocity,  # Paddle speed (normalized)
    #        opponent_x,                         # Opponent X relative to paddle
    #        opponent_y,                         # Opponent Y relative to paddle
    #        disc_x,                             # Disc X relative to paddle
    #        disc_y,                             # Disc Y relative to paddle
    #        self.disc.x_vel / self.disc.max_speed,      # Disc X velocity (normalized)
    #        self.disc.y_vel / self.disc.max_speed,      # Disc Y velocity (normalized)
    #        # Additional strategic inputs:
    #        self.disc.center_x / self.screen_width,     # Disc absolute X position
    #        # Distance to disc (normalized)
    #        min(1.0, math.sqrt((paddle.center_x - self.disc.center_x)**2 + 
    #                        (paddle.center_y - self.disc.center_y)**2) / 200),
    #        # Is disc moving toward paddle? (dot product of disc velocity and paddle-to-disc vector)
    #        self.calculate_disc_approach_factor(paddle)
    #    ]
    def get_normalized_inputs_for_paddle(self, paddle_side):
        """Get properly normalized inputs for a paddle with all positions relative to the paddle"""
        if paddle_side == 'left':
            # Blue paddle (left side)
            paddle = self.paddle1
            opponent = self.paddle2
            # Define paddle's "home" position (center of left half)
            home_x = self.screen_width / 4
            home_y = self.screen_height / 2
            
        else:
            # Red paddle (right side)
            paddle = self.paddle2
            opponent = self.paddle1
            # Define paddle's "home" position (center of right half)
            home_x = 3 * self.screen_width / 4
            home_y = self.screen_height / 2
        
        # All positions are now relative to the paddle's current position
        # Paddle position relative to its "home" position
        paddle_rel_x = (paddle.center_x - home_x) / (self.screen_width / 4)  # Normalized to [-1, 1]
        paddle_rel_y = (paddle.center_y - home_y) / (self.screen_height / 2)  # Normalized to [-1, 1]
        
        # Opponent position relative to paddle
        opponent_rel_x = (opponent.center_x - paddle.center_x) / (self.screen_width / 2)
        opponent_rel_y = (opponent.center_y - paddle.center_y) / (self.screen_height / 2)
        
        # Disc position relative to paddle
        disc_rel_x = (self.disc.center_x - paddle.center_x) / (self.screen_width / 2)
        disc_rel_y = (self.disc.center_y - paddle.center_y) / (self.screen_height / 2)
        
        # Flip x-axis for right paddle so positive x always means "toward opponent goal"
        if paddle_side == 'right':
            paddle_rel_x *= -1
            opponent_rel_x *= -1
            disc_rel_x *= -1
        
        # Distance to disc (relative, normalized)
        disc_distance = math.sqrt((paddle.center_x - self.disc.center_x)**2 + 
                                (paddle.center_y - self.disc.center_y)**2)
        disc_distance_normalized = min(1.0, disc_distance / 200)
        
        # Distance to opponent (relative, normalized)
        opponent_distance = math.sqrt((paddle.center_x - opponent.center_x)**2 + 
                                    (paddle.center_y - opponent.center_y)**2)
        opponent_distance_normalized = min(1.0, opponent_distance / 200)
        
        # Opponent's goal position relative to paddle
        if paddle_side == 'left':
            # Blue paddle attacks right goal
            opponent_goal_x = self.screen_width
            opponent_goal_y = self.screen_height / 2
        else:
            # Red paddle attacks left goal
            opponent_goal_x = 0
            opponent_goal_y = self.screen_height / 2
        
        # Opponent goal position relative to paddle
        opponent_goal_rel_x = (opponent_goal_x - paddle.center_x) / (self.screen_width / 2)
        opponent_goal_rel_y = (opponent_goal_y - paddle.center_y) / (self.screen_height / 2)
        
        # Apply x-axis flip for right paddle
        if paddle_side == 'right':
            opponent_goal_rel_x *= -1
        
        # Distance to opponent goal (relative, normalized)
        goal_distance = math.sqrt((paddle.center_x - opponent_goal_x)**2 + 
                                (paddle.center_y - opponent_goal_y)**2)
        goal_distance_normalized = min(1.0, goal_distance / 300)  # Slightly larger normalization for goal distance
        
        return [
            paddle_rel_x,                                    # Paddle X relative to home position (flipped for right)
            paddle_rel_y,                                    # Paddle Y relative to home position
            paddle.actual_speed / paddle.max_velocity,       # Paddle speed (normalized)
            opponent_rel_x,                                  # Opponent X relative to paddle (flipped for right)
            opponent_rel_y,                                  # Opponent Y relative to paddle
            disc_rel_x,                                      # Disc X relative to paddle (flipped for right)
            disc_rel_y,                                      # Disc Y relative to paddle
            self.disc.x_vel / self.disc.max_speed if paddle_side == 'left' else -self.disc.x_vel / self.disc.max_speed,  # Disc X velocity (flipped for right)
            self.disc.y_vel / self.disc.max_speed,           # Disc Y velocity (normalized)
            disc_distance_normalized,                        # Distance to disc (normalized)
            opponent_distance_normalized,                    # Distance to opponent (normalized)
            opponent_goal_rel_x,                             # Opponent goal X relative to paddle (flipped for right)
            opponent_goal_rel_y,                             # Opponent goal Y relative to paddle
            goal_distance_normalized,                        # Distance to opponent goal (normalized)
            self.calculate_disc_approach_factor(paddle, paddle_side)      # Disc approach factor
        ]


    def calculate_disc_approach_factor(self, paddle, paddle_side):
        """Calculate if disc is approaching the paddle"""
        # Vector from disc to paddle
        dx = paddle.center_x - self.disc.center_x
        dy = paddle.center_y - self.disc.center_y
        # Normalize
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0:
            return 0
        dx /= dist
        dy /= dist

        # Get disc velocity (flip x-velocity for right paddle for consistency)
        disc_x_vel = self.disc.x_vel if paddle_side == 'left' else -self.disc.x_vel
        disc_y_vel = self.disc.y_vel
        # Dot product with disc velocity (normalized)
        vel_mag = math.sqrt(disc_x_vel**2 + disc_y_vel**2)
        if vel_mag == 0:
            return 0
        # Apply coordinate flip to dx for right paddle as well
        if paddle_side == 'right':
            dx *= -1
        
        approach_factor = (dx * self.disc.x_vel + dy * self.disc.y_vel) / vel_mag
        return max(-1, min(1, approach_factor))


    def process_neat_output(self, neat_output, paddle_side):
        """Apply actions from NEAT neural networks to the paddles
        
        Args:
            neat_output: List of 4 values ["toward own goal", "toward opponent's goal", up, down] for a paddle (0-1)
            side: 'left' or 'right' to determine which paddle's controls to mock
        """
        # Convert 2-output neural network to movement commands for paddle update
        # The neural network outputs values between -1 and 1
        
        # Create a mock keys dictionary for the paddles
        # Convert to movement commands with deadzone to prevent jitter
        deadzone = 0.05
        
        x_movement = neat_output[0] if abs(neat_output[0]) > deadzone else 0
        y_movement = neat_output[1] if abs(neat_output[1]) > deadzone else 0

         # Create movement dictionary
        if paddle_side == 'left':
            # Blue paddle (WASD)
            movement = {
                pygame.K_a: x_movement < -deadzone,    # toward own goal (left)
                pygame.K_d: x_movement > deadzone,     # toward opponent's goal (right)
                pygame.K_w: y_movement < -deadzone,    # up
                pygame.K_s: y_movement > deadzone,     # down
            }
        else:
            # Red paddle (Arrow keys)
            movement = {
                #pygame.K_RIGHT: x_movement > deadzone,  # toward own goal (right)
                pygame.K_RIGHT: x_movement < -deadzone, # toward own goal (right)
                #pygame.K_LEFT: x_movement < -deadzone,  # toward opponent's goal (left)
                pygame.K_LEFT: x_movement > deadzone,  # toward opponent's goal (left)
                pygame.K_UP: y_movement < -deadzone,    # up
                pygame.K_DOWN: y_movement > deadzone,   # down
            }
        
        return movement

    # Improved fitness function
    def calculate_fitness(self, genome1, genome2, game_info):
        """
        Improved fitness calculation with better balance
        """
        blue_fitness = 0
        red_fitness = 0
        max_points = self.game.max_score

        corner_threshold = 50  # Distance from corner to be considered "in corner"
        
        blue_in_corner = (
            (self.paddle1.center_x < corner_threshold and self.paddle1.center_y < corner_threshold) or
            (self.paddle1.center_x > self.game.screen_width - corner_threshold and self.paddle1.center_y < corner_threshold) or
            (self.paddle1.center_x < corner_threshold and self.paddle1.center_y > self.game.screen_height - corner_threshold) or
            (self.paddle1.center_x > self.game.screen_width - corner_threshold and self.paddle1.center_y > self.game.screen_height - corner_threshold)
        )
        
        red_in_corner = (
            (self.paddle2.center_x < corner_threshold and self.paddle2.center_y < corner_threshold) or
            (self.paddle2.center_x > self.game.screen_width - corner_threshold and self.paddle2.center_y < corner_threshold) or
            (self.paddle2.center_x < corner_threshold and self.paddle2.center_y > self.game.screen_height - corner_threshold) or
            (self.paddle2.center_x > self.game.screen_width - corner_threshold and self.paddle2.center_y > self.game.screen_height - corner_threshold)
        )

        if (game_info['score1'] >= max_points) or (self.game.score2 >= max_points) or game_info['game_time'] > 20:
            blue_score = self.game.score1
            red_score = self.game.score2
            
            # 1. SCORING REWARDS (most important)
            blue_fitness += (blue_score / max_points) * 50.0          # Big reward for scoring
            red_fitness += (red_score / max_points) * 50.0
            
            # 2. DEFENSIVE PENALTIES (but not too harsh)
            blue_fitness -= (red_score / max_points) * 15.0            # Penalty for being scored on
            red_fitness -= (blue_score / max_points) * 15.0
    
        # 3. ACTIVE PLAY REWARDS for hitting the disc (but not if you're in a corner)
        if self.disc.check_paddle_collision(self.paddle1) and not blue_in_corner:
            blue_fitness += 1.0    # Reward hitting the puck
        if self.disc.check_paddle_collision(self.paddle2) and not red_in_corner:
            red_fitness +=  1.0    # Reward hitting the puck
    
        # 4. POSITIONING REWARDS
        # Reward being close to disc when it's on your side
        blue_fitness += self.calculate_positioning_reward(self.paddle1, 'left')
        red_fitness += self.calculate_positioning_reward(self.paddle2, 'right')
        
        # 5. ACTIVITY REWARDS (prevent staying still)
        game_time = max(1, game_info['game_time'])

        # Calculate movement ratio (lower is worse)
        blue_movement_ratio = 1.0 - (game_info['blue_paddle_ratio_times_not_moving'])
        red_movement_ratio = 1.0 - (game_info['red_paddle_ratio_times_not_moving'])
        blue_fitness += max(0, blue_movement_ratio) * 2
        red_fitness += max(0, red_movement_ratio) * 2

        # Heavy penalty for staying still
        blue_fitness -= (1.0 - blue_movement_ratio) * 10.0  # Increased penalty
        red_fitness -= (1.0 - red_movement_ratio) * 10.0

        # 6. GOAL AREA PENALTIES (prevent camping in goal)
        blue_fitness -= 0.2 * game_info['blue_paddle_num_in_goal'] / game_info['game_time']
        red_fitness -= 0.2 * game_info['red_paddle_num_in_goal'] / game_info['game_time']
        
        # Penalty for staying too close to corners and walls
        if self.paddle1.touching_wall:
            blue_fitness -= 0.75
        if self.paddle2.touching_wall:
            red_fitness -= 0.75
        if blue_in_corner:
            blue_fitness -= 0.3
        if red_in_corner:
            red_fitness -= 0.3
        
        # 7. STRATEGIC BONUSES
        # Bonus for winning quickly
        if self.game.score1 > self.game.score2:
            blue_fitness += max(0, 5.0 - game_info['game_time'] * 0.1)
        elif self.game.score2 > self.game.score1:
            red_fitness += max(0, 5.0 - game_info['game_time'] * 0.1)
        
        # 8. CONSISTENCY BONUS (reward consistent performance)
        if game_info['num_blue_hits'] > 0:
            blue_fitness += min(2.0, game_info['num_blue_hits'] * 0.5)
        if game_info['num_red_hits'] > 0:
            red_fitness += min(2.0, game_info['num_red_hits'] * 0.5)
        
        # Ensure minimum fitness
        blue_fitness = max(0.1, blue_fitness)
        red_fitness = max(0.1, red_fitness)
        
        # Assign fitness
        genome1.fitness += blue_fitness
        genome2.fitness += red_fitness

    def calculate_positioning_reward(self, paddle, side):
        """Calculate reward for good positioning"""
        reward = 0
        
        # Distance to disc (closer is better when disc is on your side)
        disc_distance = math.sqrt((paddle.center_x - self.disc.center_x)**2 + 
                                (paddle.center_y - self.disc.center_y)**2)
        
        # Check if disc is on paddle's side
        if side == 'left' and self.disc.center_x < self.screen_width/2:
            # Disc is on blue side
            reward += max(0, 1.0 - disc_distance / 200) * 2 # Reward being close
        elif side == 'right' and self.disc.center_x > self.screen_width/2:
            # Disc is on red side
            reward += max(0, 1.0 - disc_distance / 200) * 2  # Reward being close
        else:
            # Disc is on opponent's side - reward defensive positioning
            if side == 'left':
                # Blue paddle should be somewhat centered in their half
                center_x = self.screen_width / 4
            else:
                # Red paddle should be somewhat centered in their half
                center_x = 3 * self.screen_width / 4
            
            distance_from_center = abs(paddle.center_x - center_x)
            reward += max(0, 0.5 - distance_from_center / 100)
        
        return reward

def play_match(match_data):
    """
    Function to run a single match between two genomes.
    This function will be called by each worker process.
    
    Args:
        match_data: Tuple containing (genome1, genome2, config)
    
    Returns:
        Tuple of (genome1_id, genome2_id, fitness1, fitness2)
    """
    genome1, genome2, config = match_data
    
    # Create a new game instance for this match
    hockey_game = HockeyGame()
    
    # Initialize fitness values
    genome1.fitness = 0
    genome2.fitness = 0
    
    # Run the training match
    hockey_game.train_ai(genome1, genome2, config, render= False)
    
    # Return the results
    return (genome1.key, genome2.key, genome1.fitness, genome2.fitness)


def eval_genomes_parallel(genomes, config):
    """
    Parallel evaluation function that creates matches and distributes them across multiple processes.
    """
    # Prepare all matches
    matches = [] # will hold all the individual match‑up tasks to dispatch to worker processes
    # The 'genome_dict' maps genome IDs → genome objects,
    # so that after running matches we can quickly look up and update each genome’s fitness
    genome_dict = {}
    
    # Store genomes in dictionary for easy lookup
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome_dict[genome_id] = genome # Store each genome in genome_dict under its genome_id
    
    # Create all possible matches (round-robin tournament)
    # Each pair plays twice - once with each genome in each position
    for i, (genome_id1, genome1) in enumerate(genomes): # picks each genome in turn
        for genome_id2, genome2 in genomes[i+1:]: # pairs with every subsequent genome (each unique pair appears only once)
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            
            # Create copies for first match (genome1 vs genome2)
            genome1_copy_1 = neat.DefaultGenome(genome1.key)
            genome1_copy_1.configure_new(config.genome_config)
            genome1_copy_1.connections = genome1.connections.copy()
            genome1_copy_1.nodes = genome1.nodes.copy()
            
            genome2_copy_1 = neat.DefaultGenome(genome2.key)
            genome2_copy_1.configure_new(config.genome_config)
            genome2_copy_1.connections = genome2.connections.copy()
            genome2_copy_1.nodes = genome2.nodes.copy()
            
            # Create copies for second match (genome2 vs genome1)
            genome1_copy_2 = neat.DefaultGenome(genome1.key)
            genome1_copy_2.configure_new(config.genome_config)
            genome1_copy_2.connections = genome1.connections.copy()
            genome1_copy_2.nodes = genome1.nodes.copy()
            
            genome2_copy_2 = neat.DefaultGenome(genome2.key)
            genome2_copy_2.configure_new(config.genome_config)
            genome2_copy_2.connections = genome2.connections.copy()
            genome2_copy_2.nodes = genome2.nodes.copy()
            
            # Add both match combinations
            matches.append((genome1_copy_1, genome2_copy_1, config))  # genome1 left, genome2 right
            matches.append((genome2_copy_2, genome1_copy_2, config))  # genome2 left, genome1 right
    
   # Use multiprocessing to run matches in parallel
    num_processes = min(mp.cpu_count(), len(matches))  # Don't use more processes than matches
    
    print(f"Running {len(matches)} matches across {num_processes} processes...")
    #print(f"Each genome pair plays 2 matches (once in each position)")
    
    # Use tqdm progress bar to track match progress
    start_time = time.time()
    
    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm to track progress
        results = []
        with tqdm(total=len(matches), desc="Training Matches", unit="matches") as pbar:
            for result in pool.imap(play_match, matches):
                results.append(result)
                pbar.update(1)
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    num_results = len(results)
    # Aggregate average fitness results back to original genomes
    for genome1_id, genome2_id, fitness1, fitness2 in results:
        genome_dict[genome1_id].fitness += fitness1 / num_results
        genome_dict[genome2_id].fitness += fitness2 / num_results
    
    # Calculate statistics
    avg_fitness = sum(g.fitness for _, g in genomes) / len(genomes)
    best_fitness = max(g.fitness for _, g in genomes)
    
    print(f"Completed all matches in {total_time:.1f} seconds!")
    print(f"Average fitness: {avg_fitness:.2f}")
    print(f"Best fitness: {best_fitness:.2f}")

def eval_genomes_swiss(genomes, config, matches_per_genome= 6):
    """
    Swiss tournament: each genome plays against a fixed number of opponents
    """
    matches = []
    genome_dict = {}
    
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome_dict[genome_id] = genome
    
    genome_list = list(genomes)
    
    for i, (genome_id1, genome1) in enumerate(genome_list):
        # Select opponents (can be random or based on fitness)
        num_opponents = min(matches_per_genome, len(genome_list) - 1)
        
        # Random selection of opponents
        other_genomes = genome_list[:i] + genome_list[i+1:]
        opponents = random.sample(other_genomes, num_opponents)
        
        for _, opponent in opponents:
            # Create copies for first match (genome1 vs genome2)
            genome1_copy_1 = neat.DefaultGenome(genome1.key)
            genome1_copy_1.configure_new(config.genome_config)
            genome1_copy_1.connections = genome1.connections.copy()
            genome1_copy_1.nodes = genome1.nodes.copy()
            
            opponent_copy_1 = neat.DefaultGenome(opponent.key)
            opponent_copy_1.configure_new(config.genome_config)
            opponent_copy_1.connections = opponent.connections.copy()
            opponent_copy_1.nodes = opponent.nodes.copy()
            
            # Create copies for second match (genome2 vs genome1)
            genome1_copy_2 = neat.DefaultGenome(genome1.key)
            genome1_copy_2.configure_new(config.genome_config)
            genome1_copy_2.connections = genome1.connections.copy()
            genome1_copy_2.nodes = genome1.nodes.copy()
            
            opponent_copy_2 = neat.DefaultGenome(opponent.key)
            opponent_copy_2.configure_new(config.genome_config)
            opponent_copy_2.connections = opponent.connections.copy()
            opponent_copy_2.nodes = opponent.nodes.copy()

            matches.append((genome1_copy_1, opponent_copy_1, config))
            matches.append((opponent_copy_2, genome1_copy_2, config))
        # Use multiprocessing to run matches in parallel
    num_processes = min(mp.cpu_count(), len(matches))  # Don't use more processes than matches
    
    print(f"Running {len(matches)} matches across {num_processes} processes...")
    #print(f"Each genome pair plays 2 matches (once in each position)")
    
    # Use tqdm progress bar to track match progress
    start_time = time.time()
    
    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm to track progress
        results = []
        with tqdm(total=len(matches), desc="Training Matches", unit="matches") as pbar:
            for result in pool.imap(play_match, matches):
                results.append(result)
                pbar.update(1)
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    num_results = len(results)
    # Aggregate average fitness results back to original genomes
    for genome1_id, genome2_id, fitness1, fitness2 in results:
        genome_dict[genome1_id].fitness += fitness1 / num_results
        genome_dict[genome2_id].fitness += fitness2 / num_results
    
    # Calculate statistics
    avg_fitness = sum(g.fitness for _, g in genomes) / len(genomes)
    best_fitness = max(g.fitness for _, g in genomes)
    
    print(f"Completed all matches in {total_time:.1f} seconds!")
    print(f"Average fitness: {avg_fitness:.2f}")
    print(f"Best fitness: {best_fitness:.2f}")


class ProgressReporter(neat.reporting.BaseReporter):
    """Custom reporter to track overall training progress."""
    
    def __init__(self, max_generations):
        self.max_generations = max_generations
        self.generation_count = 0
        self.start_time = time.time()
    
    def start_generation(self, generation):
        self.generation_count = generation
        elapsed = time.time() - self.start_time
        if generation > 0:
            avg_time_per_gen = elapsed / generation
            remaining_time = avg_time_per_gen * (self.max_generations - generation)
            print(f"Generation {generation}/{self.max_generations} - "
                  f"Time elapsed: {elapsed:.1f}s, Estimated remaining: {remaining_time:.1f}s")
        else:
            print(f"Generation {generation}/{self.max_generations} - Starting training...")
    
    def post_evaluate(self, config, population, species, best_genome):
        generation = self.generation_count
        if population:
            avg_fitness = sum(g.fitness for g in population.values()) / len(population)
            print(f"Generation {generation} completed - Best fitness: {best_genome.fitness:.2f}, "
                  f"Average fitness: {avg_fitness:.2f}")


def run_neat(config):
    """
    Run NEAT training with parallel evaluation and progress tracking.
    """
    max_generations = 50
    
    # Create population
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    #p = neat.Population(config)
    
    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    p.add_reporter(ProgressReporter(max_generations))
    
    # Run evolution with parallel evaluation
    print(f"Starting NEAT training with {max_generations} generations...")
    print(f"Using {mp.cpu_count()} CPU cores for parallel processing\n")
    
    #winner = p.run(eval_genomes_parallel, max_generations)
    winner = p.run(eval_genomes_swiss, max_generations)
    
    
    # Save the best genome
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
        print("\n" + "="*50)
        print("TRAINING COMPLETED!")
        print("="*50)
        print(f"Saving the best genome to the pickle file 'best.pickle'")
        print(f"Best genome fitness: {winner.fitness}")
        total_time = time.time() - p.reporters.reporters[-1].start_time
        print(f"Total training time: {total_time:.2f} seconds")

def test_ai(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
        hockey_game = HockeyGame()
        hockey_game.test_ai(winner, config)


if __name__ == "__main__":
    #Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    #run_neat(config)
    test_ai(config)