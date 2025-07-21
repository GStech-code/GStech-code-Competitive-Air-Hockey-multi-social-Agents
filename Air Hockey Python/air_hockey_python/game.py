# .\venv\Scripts\Activate.ps1
# python -m air_hockey_python.game
import os
import time
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from air_hockey_python.disc import Disc
from air_hockey_python.paddle import Paddle
from air_hockey_python.helper_functions import draw_two_digit_score


class Game:
    def __init__(self, render=True):
        """Initialize the game"""
        if not render:
            # tell SDL to use the dummy video driver (no window)
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        
        # Screen dimensions
        self.screen_width = 800
        self.screen_height = 600
        
        if render:
            # only create a window when you actually want to see it
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), DOUBLEBUF|OPENGL)
            # Initialize OpenGL
            self.init_opengl()
            #pygame.display.set_caption("Air Hockey")

        
        # Game timing
        self.clock = pygame.time.Clock()
        self.last_frame_time = time.time()
        self.frame_times = []
        
        # Pause system
        self.paused = False
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.pause_pressed = False  # To prevent multiple pause toggles
        
        # Colors (RGB normalized to 0-1 for OpenGL)
        self.white = (1.0, 1.0, 1.0)
        self.blue = (0.0, 0.0, 1.0)
        self.red = (1.0, 0.0, 0.0)
        self.light_blue = (0.5, 0.5, 1.0)
        self.black = (0.0, 0.0, 0.0)
        self.gray = (0.5, 0.5, 0.5)
        self.yellow = (1.0, 1.0, 0.0)
        
        # Game state
        self.score1 = 0
        self.score2 = 0
        self.num_blue_hits = 0
        self.num_red_hits = 0
        self.serve_direction = 1
        self.game_states = []
        #self.player1_actions = []
        #self.player2_actions = []
        self.max_score = 5
        
        # Initialize game objects
        self.setup_game_objects()

    def init_opengl(self):
        """Initialize OpenGL settings for 2D rendering"""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_MULTISAMPLE)
        
        # Set up 2D orthographic projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Enable smooth rendering
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def setup_game_objects(self):
        """Initialize game objects"""
        # Goals
        goalheight = 50
        goalwidth = 10
        self.goal1 = pygame.Rect(0, self.screen_height/2 - goalheight, goalwidth, goalheight * 2)
        self.goal2 = pygame.Rect(self.screen_width - goalwidth, self.screen_height/2 - goalheight, goalwidth, goalheight * 2)
        
        # Paddles
        self.paddle1 = Paddle(
            self.screen_width / 2 - 200,
            self.screen_height / 2,
            20, 3, 'left',
            player_controlled=False # Initially player controlled
        )
        self.paddle2 = Paddle(
            self.screen_width / 2 + 200,
            self.screen_height / 2,
            20, 3, 'right',
            player_controlled=False # Initially player controlled
        )
        
        # Disc
        self.disc = Disc(
            self.screen_width / 2,
            self.screen_height / 2,
            15
        )


    def update_one_frame(self, paddle1_keys, paddle2_keys, render=False):
        """Run one frame of game logic. Optionally render for visual debugging."""
        if self.paused:
            return self.get_game_state()

        # Update paddles using NEAT decisions
        self.paddle1.update(paddle1_keys, self.screen_width, self.screen_height)
        self.paddle2.update(paddle2_keys, self.screen_width, self.screen_height)

        # Update disc movement and collisions
        self.disc.update(self.screen_width, self.screen_height)
        self.disc.check_wall_collision(self.screen_width, self.screen_height)

        # Check scoring
        score_left, score_right = self.disc.check_side_collision(
            self.screen_width, self.screen_height, self.goal1, self.goal2
        )
        if score_left:
            self.score1 += 1
            self.serve_direction = 1
            self.reset_puck()
        elif score_right:
            self.score2 += 1
            self.serve_direction = -1
            self.reset_puck()

        # Check paddle collisions
        if self.disc.check_paddle_collision(self.paddle1):
            self.disc.handle_paddle_collision(self.paddle1)
            self.num_blue_hits += 1 if self.paddle1.is_in_goal_area == False else 0
        if self.disc.check_paddle_collision(self.paddle2):
            self.disc.handle_paddle_collision(self.paddle2)
            self.num_red_hits += 1  if self.paddle2.is_in_goal_area == False else 0

        # Optional rendering
        if render:
            self.draw_field()
            self.draw_ui()

            # Draw game objects
            self.draw_circle(self.disc.center_x, self.disc.center_y, self.disc.radius, self.white)
            self.draw_circle(self.paddle1.center_x, self.paddle1.center_y, self.paddle1.radius, self.blue)
            self.draw_circle(self.paddle2.center_x, self.paddle2.center_y, self.paddle2.radius, self.red)

            pygame.display.flip()
            #self.clock.tick(60)  # Limit to 60 FPS when rendering

        return self.get_game_state()


    def toggle_pause(self):
        """Toggle pause state"""
        if self.paused:
            # Resume game
            self.paused = False
            # Add the pause duration to total pause time
            self.total_pause_time += time.time() - self.pause_start_time
            #print("Game resumed")
        else:
            # Pause game
            self.paused = True
            self.pause_start_time = time.time()
            #print("Game paused - Press P to resume")

    def draw_pause_overlay(self):
        """Draw pause overlay on screen"""
        # Semi-transparent overlay
        overlay_color = (0.0, 0.0, 0.0, 0.7)
        glColor4f(*overlay_color)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.screen_width, 0)
        glVertex2f(self.screen_width, self.screen_height)
        glVertex2f(0, self.screen_height)
        glEnd()

        
        # Draw "PAUSED" text using simple shapes
        self.draw_pause_text()
        
        # Draw instructions
        self.draw_pause_instructions()

    def draw_pause_text(self):
        """Draw 'PAUSED' text in the center"""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 - 50
        
        # Draw large "PAUSED" text using rectangles
        text_color = self.yellow
        block_size = 8
        
        # P
        self.draw_letter_p(center_x - 120, center_y, block_size, text_color)
        # A
        self.draw_letter_a(center_x - 80, center_y, block_size, text_color)
        # U
        self.draw_letter_u(center_x - 40, center_y, block_size, text_color)
        # S
        self.draw_letter_s(center_x, center_y, block_size, text_color)
        # E
        self.draw_letter_e(center_x + 40, center_y, block_size, text_color)
        # D
        self.draw_letter_d(center_x + 80, center_y, block_size, text_color)

    def draw_pause_instructions(self):
        """Draw pause instructions"""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2 + 50
        
        # Draw "Press P to Resume" using simple shapes
        text_color = self.white
        
        # This is a simplified version - you could expand this for full text
        self.draw_simple_instruction_line(center_x - 80, center_y, text_color)

    def draw_simple_instruction_line(self, x, y, color):
        """Draw a simple instruction line"""
        # Draw a simple line to indicate instructions
        self.draw_line(x, y, x + 160, y, color, 2)
        self.draw_line(x, y + 10, x + 160, y + 10, color, 2)

    def draw_letter_p(self, x, y, size, color):
        """Draw letter P using rectangles"""
        # Vertical line
        self.draw_rect(x, y, size, size * 5, color)
        # Top horizontal
        self.draw_rect(x, y, size * 3, size, color)
        # Middle horizontal
        self.draw_rect(x, y + size * 2, size * 3, size, color)
        # Right vertical (top part)
        self.draw_rect(x + size * 2, y + size, size, size, color)

    def draw_letter_a(self, x, y, size, color):
        """Draw letter A using rectangles"""
        # Left vertical
        self.draw_rect(x, y + size, size, size * 4, color)
        # Right vertical
        self.draw_rect(x + size * 2, y + size, size, size * 4, color)
        # Top horizontal
        self.draw_rect(x, y, size * 3, size, color)
        # Middle horizontal
        self.draw_rect(x, y + size * 2, size * 3, size, color)

    def draw_letter_u(self, x, y, size, color):
        """Draw letter U using rectangles"""
        # Left vertical
        self.draw_rect(x, y, size, size * 4, color)
        # Right vertical
        self.draw_rect(x + size * 2, y, size, size * 4, color)
        # Bottom horizontal
        self.draw_rect(x, y + size * 4, size * 3, size, color)

    def draw_letter_s(self, x, y, size, color):
        """Draw letter S using rectangles"""
        # Top horizontal
        self.draw_rect(x, y, size * 3, size, color)
        # Middle horizontal
        self.draw_rect(x, y + size * 2, size * 3, size, color)
        # Bottom horizontal
        self.draw_rect(x, y + size * 4, size * 3, size, color)
        # Top left vertical
        self.draw_rect(x, y + size, size, size, color)
        # Bottom right vertical
        self.draw_rect(x + size * 2, y + size * 3, size, size, color)

    def draw_letter_e(self, x, y, size, color):
        """Draw letter E using rectangles"""
        # Left vertical
        self.draw_rect(x, y, size, size * 5, color)
        # Top horizontal
        self.draw_rect(x, y, size * 3, size, color)
        # Middle horizontal
        self.draw_rect(x, y + size * 2, size * 2, size, color)
        # Bottom horizontal
        self.draw_rect(x, y + size * 4, size * 3, size, color)

    def draw_letter_d(self, x, y, size, color):
        """Draw letter D using rectangles"""
        # Left vertical
        self.draw_rect(x, y, size, size * 5, color)
        # Top horizontal
        self.draw_rect(x, y, size * 2, size, color)
        # Bottom horizontal
        self.draw_rect(x, y + size * 4, size * 2, size, color)
        # Right vertical
        self.draw_rect(x + size * 2, y + size, size, size * 3, color)

    def draw_circle(self, x, y, radius, color, filled=True):
        """Draw a circle using OpenGL"""
        glColor3f(*color)
        glBegin(GL_TRIANGLE_FAN if filled else GL_LINE_LOOP)
        
        segments = 32
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            glVertex2f(x + math.cos(angle) * radius, y + math.sin(angle) * radius)
        
        glEnd()

    def draw_line(self, x1, y1, x2, y2, color, width=1):
        """Draw a line using OpenGL"""
        glColor3f(*color)
        glLineWidth(width)
        glBegin(GL_LINES)
        glVertex2f(x1, y1)
        glVertex2f(x2, y2)
        glEnd()

    def draw_rect(self, x, y, width, height, color, filled=True):
        """Draw a rectangle using OpenGL"""
        glColor3f(*color)
        glBegin(GL_QUADS if filled else GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()

    def draw_arc(self, x, y, radius, start_angle, end_angle, color, width=1):
        """Draw an arc using OpenGL"""
        glColor3f(*color)
        glLineWidth(width)
        glBegin(GL_LINE_STRIP)
        
        segments = 32
        for i in range(segments + 1):
            angle = start_angle + (end_angle - start_angle) * i / segments
            glVertex2f(x + math.cos(angle) * radius, y + math.sin(angle) * radius)
        
        glEnd()

    def draw_field(self):
        """Draw the hockey field using OpenGL"""
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Draw goals
        self.draw_rect(self.goal1.x, self.goal1.y, self.goal1.width, self.goal1.height, self.light_blue)
        self.draw_rect(self.goal2.x, self.goal2.y, self.goal2.width, self.goal2.height, self.light_blue)
        
        # Draw center line
        self.draw_line(self.screen_width/2, 0, self.screen_width/2, self.screen_height, self.white, 5)
        
        # Draw center circle
        center_circle_radius = self.screen_width/10
        self.draw_circle(self.screen_width//2, self.screen_height//2, center_circle_radius, self.white, False)
        
        # Draw goal area semicircles
        goal_circle_radius = center_circle_radius
        
        # Left goal semicircle
        self.draw_arc(0, self.screen_height//2, goal_circle_radius, -math.pi/2, math.pi/2, self.blue, 5)
        
        # Right goal semicircle  
        self.draw_arc(self.screen_width, self.screen_height//2, goal_circle_radius, math.pi/2, 3*math.pi/2, self.red, 5)
        
        # Draw boundary lines
        goalheight = 50
        
        # Top boundaries
        self.draw_line(0, 0, self.screen_width//2 - 5, 0, self.blue, 5)
        self.draw_line(self.screen_width//2 + 5, 0, self.screen_width, 0, self.red, 5)
        
        # Bottom boundaries
        self.draw_line(0, self.screen_height, self.screen_width//2 - 5, self.screen_height, self.blue, 5)
        self.draw_line(self.screen_width//2 + 5, self.screen_height, self.screen_width, self.screen_height, self.red, 5)
        
        # Left side boundaries
        self.draw_line(0, 0, 0, self.screen_height//2 - goalheight, self.blue, 5)
        self.draw_line(0, self.screen_height//2 + goalheight, 0, self.screen_height, self.blue, 5)
        
        # Right side boundaries
        self.draw_line(self.screen_width, 0, self.screen_width, self.screen_height//2 - goalheight, self.red, 5)
        self.draw_line(self.screen_width, self.screen_height//2 + goalheight, self.screen_width, self.screen_height, self.red, 5)

    def draw_ui(self):
        """Draw UI elements (score, etc.)"""
        # Draw score indicators using simple rectangles
        # Left side score (blue)
        score_width = 20
        score_height = 10
        score_y = 20
        
        # Draw blue score as rectangles
        for i in range(min(self.score1, 10)):  # Cap visual indicators at 10
            x = 50 + i * (score_width + 5)
            self.draw_rect(x, score_y, score_width, score_height, self.blue)
        
        # Draw red score as rectangles
        for i in range(min(self.score2, 10)):  # Cap visual indicators at 10
            x = self.screen_width - 50 - (i + 1) * (score_width + 5)
            self.draw_rect(x, score_y, score_width, score_height, self.red)
        
        # Draw center divider for score area
        self.draw_line(self.screen_width/2, 10, self.screen_width/2, 50, self.white, 2)
        
        # Draw 2-digit score displays
        draw_two_digit_score(self, self.score1, 25, 60, self.blue)
        draw_two_digit_score(self, self.score2, self.screen_width - 70, 60, self.red)


    def get_game_state(self):
        """Get current game state for neural network (normelized)"""
        game_state = {
            # Normalize paddle position to [-1, 1] for x, [0, 1] for y
            'blue_paddle_actual_speed': self.paddle1.actual_speed / abs(self.paddle1.max_velocity),
            'blue_paddle_x': (self.paddle1.center_x - self.screen_width/4) / (self.screen_width/4),
            'blue_paddle_y': self.paddle1.center_y / self.screen_height,
            'blue_paddle_ratio_times_not_moving': self.paddle1.times_not_moving / ((pygame.time.get_ticks() / 1000.0) - self.total_pause_time),
            'red_paddle_x': (self.paddle2.center_x - 3*self.screen_width/4) / (self.screen_width/4),
            'red_paddle_y': self.paddle2.center_y / self.screen_height,
            'red_paddle_ratio_times_not_moving': self.paddle2.times_not_moving / ((pygame.time.get_ticks() / 1000.0) - self.total_pause_time),
            'red_paddle_actual_speed': self.paddle2.actual_speed / abs(self.paddle2.max_velocity),
            'disc_x': self.disc.center_x / self.screen_width,
            'disc_y': self.disc.center_y / self.screen_height,
            'disc_velocity_x': self.disc.x_vel / self.disc.max_speed,
            'disc_velocity_y': self.disc.y_vel / self.disc.max_speed,
        #    'blue_paddle_x_raw': self.paddle1.center_x,
        #    'blue_paddle_y_raw': self.paddle1.center_y,
        #    'red_paddle_x_raw': self.paddle2.center_x,
        #    'red_paddle_y_raw': self.paddle2.center_y,
        #    'disc_x_raw': self.disc.center_x,
        #    'disc_y_raw': self.disc.center_y,
        #    'disc_velocity_x_raw': self.disc.x_vel,
        #    'disc_velocity_y_raw': self.disc.y_vel,
            'blue_paddle_in_own_goal': self.paddle1.is_in_goal_area(self.screen_width, self.screen_height),
            'red_paddle_in_own_goal': self.paddle2.is_in_goal_area(self.screen_width, self.screen_height),
            'blue_paddle_num_in_goal': self.paddle1.num_in_goal,
            'red_paddle_num_in_goal': self.paddle2.num_in_goal,
            'score1': self.score1,
            'score2': self.score2,
        #    'serve_direction': self.serve_direction,
            'blue_paddle_to_disc_distance': math.sqrt((self.paddle1.center_x - self.disc.center_x)**2 + (self.paddle1.center_y - self.disc.center_y)**2),
            'red_paddle_to_disc_distance': math.sqrt((self.paddle2.center_x - self.disc.center_x)**2 + (self.paddle2.center_y - self.disc.center_y)**2),
            'game_time': (pygame.time.get_ticks() / 1000.0) - self.total_pause_time,
            'num_blue_hits': self.num_blue_hits,
            'num_red_hits': self.num_red_hits,
        #    'paused': self.paused
        }
        return game_state
    

    def reset_puck(self):
        """Reset puck after scoring"""
        self.disc.reset(self.screen_width, self.screen_height, self.serve_direction)
        #print(f"Score: Player 1: {self.score1}, Player 2: {self.score2}")
        time.sleep(0.5)

    def reset_game(self):
        """Reset the entire game"""
        self.paddle1.center_x = self.screen_width / 2 - 200
        self.paddle1.center_y = self.screen_height / 2
        self.paddle1.velocity = 0
        self.paddle2.center_x = self.screen_width / 2 + 200
        self.paddle2.center_y = self.screen_height / 2
        self.paddle2.velocity = 0
        self.score1 = 0
        self.score2 = 0
        self.serve_direction = 1
        self.game_states = []
        self.paused = False
        self.total_pause_time = 0
        self.num_blue_hits = 0
        self.num_red_hits = 0
        self.reset_puck()

    def update_game_state(self, keys):
        """Update all game objects with improved responsiveness"""
        if self.paused:
            return  # Don't update game state when paused
            
        # Update paddles with higher frequency
        self.paddle1.update(keys, self.screen_width, self.screen_height)

        self.paddle2.update(keys, self.screen_width, self.screen_height)
        
        # Update disc with smoother physics
        self.disc.update(self.screen_width, self.screen_height)
        
        # Check collisions
        self.disc.check_wall_collision(self.screen_width, self.screen_height)
        
        # Check scoring
        score_left, score_right = self.disc.check_side_collision(self.screen_width, self.screen_height, self.goal1, self.goal2)
        if score_left:
            self.score1 += 1
            self.serve_direction = 1
            self.reset_puck()
        elif score_right:
            self.score2 += 1
            self.serve_direction = -1
            self.reset_puck()
        
        # Check paddle collisions
        if self.disc.check_paddle_collision(self.paddle1):
            self.disc.handle_paddle_collision(self.paddle1)
        if self.disc.check_paddle_collision(self.paddle2):
            self.disc.handle_paddle_collision(self.paddle2)


    def run(self):
        """Main game loop with GPU acceleration"""
        game_exit = False
        frame_count = 0
        #print("Starting GPU-accelerated Air Hockey!")
        #print("Controls:")
        #print("Player 1 (Blue): WASD keys")
        #print("Player 2 (Red): Arrow keys")
        #print("P: Pause/Resume")
        #print("R: Reset game")
        #print("ESC: Exit game")
        
        # Now that OpenGL context is created, we can query OpenGL info
        #try:
            #print("OpenGL Version:", glGetString(GL_VERSION).decode())
            #print("Graphics Card:", glGetString(GL_RENDERER).decode())
        #except Exception as e:
            #print("Could not get OpenGL info:", e)
        
        try:
            while not game_exit:
                # Handle events
                for event in pygame.event.get():
                    if self.score1 > self.max_score or self.score2 > self.max_score:
                        game_exit = True
                    if event.type == pygame.QUIT:
                        game_exit = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            game_exit = True
                        if event.key == pygame.K_r:
                            self.reset_game()
                        if event.key == pygame.K_p:
                            if not self.pause_pressed:
                                self.toggle_pause()
                                self.pause_pressed = True
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_p:
                            self.pause_pressed = False
                
                # Get input with reduced latency
                keys = pygame.key.get_pressed()
                # Only update game state if not paused
                if not self.paused:
                    # Collect training data
                    current_state = self.get_game_state()
                    
                    # Update game state
                    self.update_game_state(keys)
                
                # Always draw the game (even when paused)
                self.draw_field()
                self.draw_ui()
                
                # Draw game objects
                # Draw the Disc:
                self.draw_circle(self.disc.center_x, self.disc.center_y, self.disc.radius, self.white)
                # Draw the Blue Player:
                self.draw_circle(self.paddle1.center_x, self.paddle1.center_y, self.paddle1.radius, self.blue)
                # Draw the Red Player:
                self.draw_circle(self.paddle2.center_x, self.paddle2.center_y, self.paddle2.radius, self.red)
                
                # Draw pause overlay if paused
                if self.paused:
                    self.draw_pause_overlay()
                
                # Update display
                pygame.display.flip()
                
                # Control frame rate - increased for better responsiveness
                self.clock.tick(120)  # 120 FPS for ultra-smooth gameplay
                frame_count += 1

        finally:
            # Clean up textures before quitting
            #self.cleanup_textures()
            print(f"Game ended. Collected {len(self.game_states)} training samples.")
            pygame.quit()


# Initialize and run the game
if __name__ == "__main__":
    game = Game()
    game.run()
