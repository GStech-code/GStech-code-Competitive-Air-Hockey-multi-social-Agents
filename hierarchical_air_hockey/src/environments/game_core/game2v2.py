import os
import time
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from .disc import Disc
from .paddle import Paddle
from .helper_functions import draw_two_digit_score


class Game2v2:
    def __init__(self, render=True):
        """Initialize the 2v2 game"""
        if not render:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        
        # Screen dimensions
        self.screen_width = 800
        self.screen_height = 600
        
        if render:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), DOUBLEBUF|OPENGL)
            self.init_opengl()

        # Game timing
        self.clock = pygame.time.Clock()
        self.last_frame_time = time.time()
        self.frame_times = []
        
        # Pause system
        self.paused = False
        self.pause_start_time = 0
        self.total_pause_time = 0
        self.pause_pressed = False
        
        # Colors (RGB normalized to 0-1 for OpenGL)
        self.white = (1.0, 1.0, 1.0)
        self.blue = (0.0, 0.0, 1.0)
        self.red = (1.0, 0.0, 0.0)
        self.light_blue = (0.3, 0.3, 1.0)
        self.dark_blue = (0.0, 0.0, 0.7)
        self.light_red = (1.0, 0.3, 0.3)
        self.dark_red = (0.7, 0.0, 0.0)
        self.black = (0.0, 0.0, 0.0)
        self.gray = (0.5, 0.5, 0.5)
        self.yellow = (1.0, 1.0, 0.0)
        
        # Game state
        self.score1 = 0  # Blue team score
        self.score2 = 0  # Red team score
        self.num_blue_hits = 0
        self.num_red_hits = 0
        self.serve_direction = 1
        self.game_states = []
        self.max_score = 5
        
        # Initialize game objects
        self.setup_game_objects()

    def init_opengl(self):
        """Initialize OpenGL settings for 2D rendering"""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_MULTISAMPLE)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    def setup_game_objects(self):
        """Initialize game objects for 2v2"""
        # Goals
        goalheight = 50
        goalwidth = 10
        self.goal1 = pygame.Rect(0, self.screen_height/2 - goalheight, goalwidth, goalheight * 2)
        self.goal2 = pygame.Rect(self.screen_width - goalwidth, self.screen_height/2 - goalheight, goalwidth, goalheight * 2)
        self.paddle1A = Paddle(self.screen_width / 6, self.screen_height / 2, 20, 3, 'left') 
        self.paddle1B = Paddle(2 * self.screen_width / 6, self.screen_height / 2, 20, 3, 'left') 
        self.paddle2A = Paddle(5 * self.screen_width / 6, self.screen_height / 2, 20, 3, 'right') 
        self.paddle2B = Paddle(4 * self.screen_width / 6, self.screen_height / 2, 20, 3, 'right')

        # Store paddles in lists for easier iteration
        self.blue_paddles = [self.paddle1A, self.paddle1B]
        self.red_paddles = [self.paddle2A, self.paddle2B]
        self.all_paddles = self.blue_paddles + self.red_paddles

        # Disc
        self.disc = Disc(
            self.screen_width / 2,
            self.screen_height / 2,
            15
        )
        
    def update_one_frame(self, blue_decisions, red_decisions, render=False):
        """Run one frame of game logic for 2v2
        
        Args:
            blue_decisions: List of 2 decision dictionaries for blue team paddles
            red_decisions: List of 2 decision dictionaries for red team paddles
        """
        if self.paused:
            return self.get_game_state()

        # Update blue team paddles
        for i, paddle in enumerate(self.blue_paddles):
            if i < len(blue_decisions):
                paddle.update(blue_decisions[i], self.screen_width, self.screen_height)
        
        # Update red team paddles
        for i, paddle in enumerate(self.red_paddles):
            if i < len(red_decisions):
                paddle.update(red_decisions[i], self.screen_width, self.screen_height)

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

        # Check paddle collisions for all paddles
        for paddle in self.blue_paddles:
            if self.disc.check_paddle_collision(paddle):
                self.disc.handle_paddle_collision(paddle)
                if not paddle.is_in_goal_area(self.screen_width, self.screen_height):
                    self.num_blue_hits += 1
        
        for paddle in self.red_paddles:
            if self.disc.check_paddle_collision(paddle):
                self.disc.handle_paddle_collision(paddle)
                if not paddle.is_in_goal_area(self.screen_width, self.screen_height):
                    self.num_red_hits += 1

        # Optional rendering
        if render:
            self.draw_field()
            self.draw_ui()
            self.draw_game_objects()
            pygame.display.flip()

        return self.get_game_state()

    def draw_game_objects(self):
        """Draw all game objects"""
        # Draw disc
        self.draw_circle(self.disc.center_x, self.disc.center_y, self.disc.radius, self.white)
        
        # Draw blue team paddles (different shades)
        self.draw_circle(self.paddle1A.center_x, self.paddle1A.center_y, 
                        self.paddle1A.radius, self.blue)
        self.draw_circle(self.paddle1B.center_x, self.paddle1B.center_y, 
                        self.paddle1B.radius, self.light_blue)
        
        # Draw red team paddles (different shades)
        self.draw_circle(self.paddle2A.center_x, self.paddle2A.center_y, 
                        self.paddle2A.radius, self.red)
        self.draw_circle(self.paddle2B.center_x, self.paddle2B.center_y, 
                        self.paddle2B.radius, self.light_red)


    def get_game_state(self):
        """Get current game state for neural network (normelized)"""
        game_state = {
            # Normalize paddle position to [-1, 1] for x, [0, 1] for y
            'blueA_paddle_actual_speed': self.paddle1A.actual_speed / abs(self.paddle1A.max_velocity),
            'blueB_paddle_actual_speed': self.paddle1B.actual_speed / abs(self.paddle1B.max_velocity),
            'blueA_paddle_x': (self.paddle1A.center_x - self.screen_width/4) / (self.screen_width/4),
            'blueB_paddle_x': (self.paddle1B.center_x - self.screen_width/4) / (self.screen_width/4),
            'blueA_paddle_y': self.paddle1A.center_y / self.screen_height,
            'blueB_paddle_y': self.paddle1B.center_y / self.screen_height,
            'blueA_paddle_ratio_times_not_moving': self.paddle1A.times_not_moving / ((pygame.time.get_ticks() / 1000.0) - self.total_pause_time),
            'blueB_paddle_ratio_times_not_moving': self.paddle1B.times_not_moving / ((pygame.time.get_ticks() / 1000.0) - self.total_pause_time),
            'redA_paddle_x': (self.paddle2A.center_x - 3*self.screen_width/4) / (self.screen_width/4),
            'redB_paddle_x': (self.paddle2B.center_x - 3*self.screen_width/4) / (self.screen_width/4),
            'redA_paddle_y': self.paddle2A.center_y / self.screen_height,
            'redB_paddle_y': self.paddle2B.center_y / self.screen_height,
            'redA_paddle_ratio_times_not_moving': self.paddle2A.times_not_moving / ((pygame.time.get_ticks() / 1000.0) - self.total_pause_time),
            'redB_paddle_ratio_times_not_moving': self.paddle2B.times_not_moving / ((pygame.time.get_ticks() / 1000.0) - self.total_pause_time),
            'redA_paddle_actual_speed': self.paddle2A.actual_speed / abs(self.paddle2A.max_velocity),
            'redB_paddle_actual_speed': self.paddle2B.actual_speed / abs(self.paddle2B.max_velocity),
            'disc_x': self.disc.center_x / self.screen_width,
            'disc_y': self.disc.center_y / self.screen_height,
            'disc_velocity_x': self.disc.x_vel / self.disc.max_speed,
            'disc_velocity_y': self.disc.y_vel / self.disc.max_speed,
            'blueA_paddle_in_own_goal': self.paddle1A.is_in_goal_area(self.screen_width, self.screen_height),
            'blueB_paddle_in_own_goal': self.paddle1B.is_in_goal_area(self.screen_width, self.screen_height),
            'redA_paddle_in_own_goal': self.paddle2A.is_in_goal_area(self.screen_width, self.screen_height),
            'redB_paddle_in_own_goal': self.paddle2B.is_in_goal_area(self.screen_width, self.screen_height),
            'blueA_paddle_num_in_goal': self.paddle1A.num_in_goal,
            'blueB_paddle_num_in_goal': self.paddle1B.num_in_goal,
            'redA_paddle_num_in_goal': self.paddle2A.num_in_goal,
            'redB_paddle_num_in_goal': self.paddle2B.num_in_goal,
            'score1': self.score1,
            'score2': self.score2,
            'blueA_paddle_to_disc_distance': math.sqrt((self.paddle1A.center_x - self.disc.center_x)**2 + (self.paddle1A.center_y - self.disc.center_y)**2),
            'blueB_paddle_to_disc_distance': math.sqrt((self.paddle1B.center_x - self.disc.center_x)**2 + (self.paddle1B.center_y - self.disc.center_y)**2),
            'redA_paddle_to_disc_distance': math.sqrt((self.paddle2A.center_x - self.disc.center_x)**2 + (self.paddle2A.center_y - self.disc.center_y)**2),
            'redB_paddle_to_disc_distance': math.sqrt((self.paddle2B.center_x - self.disc.center_x)**2 + (self.paddle2B.center_y - self.disc.center_y)**2),
            'game_time': (pygame.time.get_ticks() / 1000.0) - self.total_pause_time,
            'num_blue_hits': self.num_blue_hits,
            'num_red_hits': self.num_red_hits,
        }
        return game_state

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

    def reset_game(self):
        """Reset the entire game state"""
        self.score1 = 0
        self.score2 = 0
        self.num_blue_hits = 0
        self.num_red_hits = 0
        self.serve_direction = 1
        self.total_pause_time = 0
        self.paused = False
        
        # Reset paddle positions
        self.paddle1A.center_x = self.screen_width / 6
        self.paddle1A.center_y = self.screen_height / 2
        self.paddle1B.center_x = 2 * self.screen_width / 6
        self.paddle1B.center_y = self.screen_height / 2
        self.paddle2A.center_x = 5 * self.screen_width / 6
        self.paddle2A.center_y = self.screen_height / 2
        self.paddle2B.center_x = 4 * self.screen_width / 6
        self.paddle2B.center_y = self.screen_height / 2
        
        # Reset paddle states
        for paddle in self.all_paddles:
            paddle.reset_position(paddle.center_x, paddle.center_y)
        
        self.reset_puck()

    def reset_puck(self):
        """Reset puck to center with random direction"""
        self.disc.center_x = self.screen_width / 2
        self.disc.center_y = self.screen_height / 2
        
        # Random serving direction
        import random
        angle = random.uniform(-math.pi/4, math.pi/4)  # Random angle within 45 degrees
        speed = 3.0
        
        self.disc.x_vel = speed * math.cos(angle) * self.serve_direction
        self.disc.y_vel = speed * math.sin(angle)

