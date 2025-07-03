import pygame
import time
import math
import random
from pygame.locals import *

class Disc:
    def __init__(self, x, y, radius, image_path):
        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.max_speed = 15
        angle = self._get_random_angle(-30, 30, [0])
        self.x_vel = abs(math.cos(angle) * self.max_speed)
        self.y_vel = math.sin(angle) * self.max_speed

        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (radius * 2, radius * 2))
    
    def _get_random_angle(self, min_angle, max_angle, excluded):
        angle = 0
        while angle in excluded:
            angle = math.radians(random.randrange(min_angle, max_angle))

        return angle

    def update(self, screen_width, screen_height):
        """Update disc position based on velocity"""
        self.center_x += self.x_vel
        self.center_y += self.y_vel
    
    def check_wall_collision(self, screen_width, screen_height):
        """Check and handle wall collisions"""
        # Top and bottom walls
        if self.center_y - self.radius < 0 or self.center_y + self.radius > screen_height:
            self.y_vel *= -1
            # Prevent sticking to walls
            if self.center_y - self.radius < 0:
                self.center_y = self.radius
            else:
                self.center_y = screen_height - self.radius
    
    def check_side_collision(self, screen_width, screen_height, goal1, goal2):
        """Check side collisions and goal scoring"""
        score_left = 0
        score_right = 0
        
        # Left wall
        if self.center_x - self.radius < 0:
            # Check if it's within goal height
            if self.center_y > goal1.top and self.center_y < goal1.bottom:
                score_right = 1
            else:
                self.x_vel *= -1
                self.center_x = self.radius
        
        # Right wall
        if self.center_x + self.radius > screen_width:
            # Check if it's within goal height
            if self.center_y > goal2.top and self.center_y < goal2.bottom:
                score_left = 1
            else:
                self.x_vel *= -1
                self.center_x = screen_width - self.radius
        
        return score_left, score_right
    
    def check_paddle_collision(self, paddle):
        """Check collision with a paddle"""
        distance = math.sqrt((self.center_x - paddle.center_x)**2 + (self.center_y - paddle.center_y)**2)
        return distance <= (self.radius + paddle.radius)
    
    def handle_paddle_collision(self, paddle):
        """Handle collision with a paddle using realistic circular collision physics"""
        # Calculate collision vector (from paddle center to disc center)
        dx = self.center_x - paddle.center_x
        dy = self.center_y - paddle.center_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Avoid division by zero
        if distance == 0:
            dx, dy = 1, 0
            distance = 1
        
        # Normalize collision vector
        nx = dx / distance  # Normal vector x
        ny = dy / distance  # Normal vector y
        
        # Relative velocity (disc velocity minus paddle velocity)
        dvx = self.x_vel - paddle.actual_velocity_x
        dvy = self.y_vel - paddle.actual_velocity_y
        
        # Relative velocity along collision normal
        dvn = dvx * nx + dvy * ny
        
        # Don't resolve if objects are separating
        if dvn > 0:
            return
        
        # Collision impulse with realistic physics
        # Mass ratio (assuming disc is lighter than paddle)
        disc_mass = 1.0
        paddle_mass = 3.0
        mass_ratio = (2 * paddle_mass) / (disc_mass + paddle_mass)
        
        # Restitution coefficient (energy loss)
        restitution = 0.85  # Slightly less than perfectly elastic for realistic feel
        impulse = -(1 + restitution) * dvn * mass_ratio
        
        # Apply impulse to disc velocity
        self.x_vel += impulse * nx
        self.y_vel += impulse * ny
        
        # Add paddle velocity influence (paddle can "push" the disc)
        velocity_transfer = 0.3  # How much paddle velocity affects disc
        self.x_vel += paddle.actual_velocity_x * velocity_transfer
        self.y_vel += paddle.actual_velocity_y * velocity_transfer
        
        # Separate objects to prevent sticking
        overlap = (self.radius + paddle.radius) - distance
        if overlap > 0:
            # Move disc out of paddle
            separation = overlap + 1  # Add small buffer
            self.center_x += nx * separation
            self.center_y += ny * separation
        
        # Apply velocity limits to prevent unrealistic speeds
        max_speed = self.max_speed
        current_speed = math.sqrt(self.x_vel**2 + self.y_vel**2)
        if current_speed > max_speed:
            self.x_vel = (self.x_vel / current_speed) * max_speed
            self.y_vel = (self.y_vel / current_speed) * max_speed
        
        # Add slight randomness for more dynamic gameplay
        import random
        self.x_vel += random.uniform(-0.2, 0.2)
        self.y_vel += random.uniform(-0.2, 0.2)
    
    def reset(self, screen_width, screen_height, serve_direction):
        """Reset disc to center with serve direction"""
        self.center_x = screen_width / 2
        self.center_y = screen_height / 2
        angle = self._get_random_angle(-30, 30, [0])
        self.x_vel = abs(math.cos(angle) * 7) * serve_direction
        self.y_vel = math.sin(angle) * 7 * serve_direction
        
    def draw(self, screen):
        """Draw the disc"""
        disc_rect = pygame.Rect(
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.radius * 2,
            self.radius * 2
        )
        screen.blit(self.image, disc_rect.topleft)

class Paddle:
    def __init__(self, x, y, radius, velocity, image_path, side='left'):
        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.velocity = velocity
        self.side = side  # 'left' or 'right'
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (radius * 2, radius * 2))
        self.color = (255, 100, 100) if side == 'left' else (20, 20, 100)
        
        # Track previous position for velocity calculation
        self.prev_x = x
        self.prev_y = y
        self.actual_velocity_x = 0
        self.actual_velocity_y = 0
    
    def update(self, keys, screen_width, screen_height):
        """Update paddle position based on input"""
        # Store previous position
        self.prev_x = self.center_x
        self.prev_y = self.center_y
        
        if self.side == 'left':
            # WASD controls
            if keys[K_a]:
                self.center_x -= self.velocity
            if keys[K_d]:
                self.center_x += self.velocity
            if keys[K_w]:
                self.center_y -= self.velocity
            if keys[K_s]:
                self.center_y += self.velocity
        else:
            # Arrow key controls
            if keys[K_LEFT]:
                self.center_x -= self.velocity
            if keys[K_RIGHT]:
                self.center_x += self.velocity
            if keys[K_UP]:
                self.center_y -= self.velocity
            if keys[K_DOWN]:
                self.center_y += self.velocity
        
        # Apply boundaries
        self.apply_boundaries(screen_width, screen_height)
        
        # Calculate actual velocity (how much the paddle actually moved)
        self.actual_velocity_x = self.center_x - self.prev_x
        self.actual_velocity_y = self.center_y - self.prev_y
    
    def apply_boundaries(self, screen_width, screen_height):
        """Apply movement boundaries"""
        # Top and bottom boundaries
        if self.center_y - self.radius < 0:
            self.center_y = self.radius
        elif self.center_y + self.radius > screen_height:
            self.center_y = screen_height - self.radius
        
        # Side boundaries
        if self.side == 'left':
            # Left paddle boundaries
            if self.center_x - self.radius < 0:
                self.center_x = self.radius
            elif self.center_x + self.radius > screen_width / 2:
                self.center_x = screen_width / 2 - self.radius
        else:
            # Right paddle boundaries
            if self.center_x + self.radius > screen_width:
                self.center_x = screen_width - self.radius
            elif self.center_x - self.radius < screen_width / 2:
                self.center_x = screen_width / 2 + self.radius
    
    def get_action_vector(self, keys):
        """Get action vector for neural network training"""
        action = [0, 0, 0, 0]  # [left, right, up, down]
        
        if self.side == 'left':
            if keys[K_a]:
                action[0] = 1
            if keys[K_d]:
                action[1] = 1
            if keys[K_w]:
                action[2] = 1
            if keys[K_s]:
                action[3] = 1
        else:
            if keys[K_LEFT]:
                action[0] = 1
            if keys[K_RIGHT]:
                action[1] = 1
            if keys[K_UP]:
                action[2] = 1
            if keys[K_DOWN]:
                action[3] = 1
        
        return action
    
    def is_in_goal_area(self, screen_width, screen_height):
        """Check if paddle is in goal area"""
        goal_circle_radius = screen_width / 10
        
        if self.side == 'left':
            goal_center = (0, screen_height / 2)
            distance = math.sqrt((self.center_x - goal_center[0])**2 + (self.center_y - goal_center[1])**2)
            return distance <= goal_circle_radius and self.center_x >= 0
        else:
            goal_center = (screen_width, screen_height / 2)
            distance = math.sqrt((self.center_x - goal_center[0])**2 + (self.center_y - goal_center[1])**2)
            return distance <= goal_circle_radius and self.center_x <= screen_width
    
    def draw(self, screen):
        """Draw the paddle"""
        # Draw circle
        pygame.draw.circle(screen, self.color, (int(self.center_x), int(self.center_y)), self.radius)
        
        # Draw image
        paddle_rect = pygame.Rect(
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.radius * 2,
            self.radius * 2
        )
        screen.blit(self.image, paddle_rect.topleft)

class Game:
    def __init__(self):
        pygame.init()
        
        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.green = (0, 150, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.light_blue = (147, 251, 253)
        
        # Game setup
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Air Hockey!')
        
        # Fonts
        self.smallfont = pygame.font.SysFont("comicsansms", 25)
        self.medfont = pygame.font.SysFont("comicsansms", 45)
        self.largefont = pygame.font.SysFont("comicsansms", 65)
        
        # Game objects
        self.setup_game_objects()
        
        # Game state
        self.score1 = 0
        self.score2 = 0
        self.serve_direction = 1
        
        # Training data
        self.game_states = []
        self.player1_actions = []
        self.player2_actions = []
    
    def setup_game_objects(self):
        """Initialize game objects"""
        # Goals
        goalheight = 50
        goalwidth = 10
        self.goal1 = pygame.Rect(0, self.screen.get_height()/2 - goalheight, goalwidth, goalheight * 2)
        self.goal2 = pygame.Rect(self.screen.get_width() - goalwidth, self.screen.get_height()/2 - goalheight, goalwidth, goalheight * 2)
        
        # Paddles
        self.paddle1 = Paddle(
            self.screen.get_width() / 2 - 200,
            self.screen.get_height() / 2,
            20, 8, './Air-Hockey-python/bluepad.png', 'left'
        )
        self.paddle2 = Paddle(
            self.screen.get_width() / 2 + 200,
            self.screen.get_height() / 2,
            20, 8, './Air-Hockey-python/redpad.png', 'right'
        )
        
        # Disc
        self.disc = Disc(
            self.screen.get_width() / 2,
            self.screen.get_height() / 2,
            15, './Air-Hockey-python/disc.png'
        )
    
    def get_game_state(self):
        """Get current game state for neural network"""
        game_state = {
            'blue_paddle_x': self.paddle1.center_x / self.screen.get_width(),
            'blue_paddle_y': self.paddle1.center_y / self.screen.get_height(),
            'red_paddle_x': self.paddle2.center_x / self.screen.get_width(),
            'red_paddle_y': self.paddle2.center_y / self.screen.get_height(),
            'disc_x': self.disc.center_x / self.screen.get_width(),
            'disc_y': self.disc.center_y / self.screen.get_height(),
            'disc_velocity_x': self.disc.x_vel / 10.0,
            'disc_velocity_y': self.disc.y_vel / 10.0,
            'blue_paddle_x_raw': self.paddle1.center_x,
            'blue_paddle_y_raw': self.paddle1.center_y,
            'red_paddle_x_raw': self.paddle2.center_x,
            'red_paddle_y_raw': self.paddle2.center_y,
            'disc_x_raw': self.disc.center_x,
            'disc_y_raw': self.disc.center_y,
            'disc_velocity_x_raw': self.disc.x_vel,
            'disc_velocity_y_raw': self.disc.y_vel,
            'blue_paddle_in_own_goal': self.paddle1.is_in_goal_area(self.screen.get_width(), self.screen.get_height()),
            'red_paddle_in_own_goal': self.paddle2.is_in_goal_area(self.screen.get_width(), self.screen.get_height()),
            'score1': self.score1,
            'score2': self.score2,
            'serve_direction': self.serve_direction,
            'blue_paddle_to_disc_distance': math.sqrt((self.paddle1.center_x - self.disc.center_x)**2 + (self.paddle1.center_y - self.disc.center_y)**2),
            'red_paddle_to_disc_distance': math.sqrt((self.paddle2.center_x - self.disc.center_x)**2 + (self.paddle2.center_y - self.disc.center_y)**2),
            'game_time': pygame.time.get_ticks() / 1000.0
        }
        return game_state
    
    def get_neural_network_input_vector(self):
        """Get input vector for neural network"""
        state = self.get_game_state()
        return [
            state['blue_paddle_x'],
            state['blue_paddle_y'],
            state['red_paddle_x'],
            state['red_paddle_y'],
            state['disc_x'],
            state['disc_y'],
            state['disc_velocity_x'],
            state['disc_velocity_y'],
        ]
    
    def reset_puck(self):
        """Reset puck after scoring"""
        self.disc.reset(self.screen.get_width(), self.screen.get_height(), self.serve_direction)
        print(f"Score: Player 1: {self.score1}, Player 2: {self.score2}")
        time.sleep(0.5)
    
    def text_objects(self, text, color, size):
        """Create text objects"""
        if size == "small":
            textSurface = self.smallfont.render(text, True, color)
        elif size == "medium":
            textSurface = self.medfont.render(text, True, color)
        elif size == "large":
            textSurface = self.largefont.render(text, True, color)
        return textSurface, textSurface.get_rect()
    
    def message_to_screen(self, msg, color, y_displace=0, x_displace=0, size="small"):
        """Display message on screen"""
        textSurf, textRect = self.text_objects(msg, color, size)
        textRect.center = (self.screen.get_width()/2+x_displace, (self.screen.get_height()/2) + y_displace)
        self.screen.blit(textSurf, textRect)
    
    def pause(self):
        """Pause game"""
        paused = True
        self.message_to_screen("Paused", self.light_blue, -100, size="large")
        self.message_to_screen("Press c to continue , q to quit", self.light_blue, 25)
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
            self.clock.tick(5)
    
    def draw_field(self):
        """Draw the hockey field"""
        # Fill background
        self.screen.fill(self.black)
        
        # Draw goals
        pygame.draw.rect(self.screen, self.light_blue, self.goal1)
        pygame.draw.rect(self.screen, self.light_blue, self.goal2)
        
        # Draw center line and circle
        divline1 = self.screen.get_width()/2, 0
        divline2 = self.screen.get_width()/2, self.screen.get_height()
        pygame.draw.line(self.screen, self.white, divline1, divline2, 5)
        
        center_circle_radius = self.screen.get_width()/10
        pygame.draw.circle(self.screen, self.white, (self.screen.get_width()//2, self.screen.get_height()//2), center_circle_radius, 5)
        
        # Draw goal area semicircles
        goal_circle_radius = center_circle_radius
        
        # Left goal semicircle
        left_goal_rect = pygame.Rect(
            -goal_circle_radius,
            self.screen.get_height()//2 - goal_circle_radius,
            goal_circle_radius * 2,
            goal_circle_radius * 2
        )
        pygame.draw.arc(self.screen, self.blue, left_goal_rect, -math.pi/2, math.pi/2, 5)
        
        # Right goal semicircle
        right_goal_rect = pygame.Rect(
            self.screen.get_width() - goal_circle_radius,
            self.screen.get_height()//2 - goal_circle_radius,
            goal_circle_radius * 2,
            goal_circle_radius * 2
        )
        pygame.draw.arc(self.screen, self.red, right_goal_rect, math.pi/2, 3*math.pi/2, 5)
        
        # Draw boundary lines
        goalheight = 50
        pygame.draw.line(self.screen, self.blue, (0, 0), (self.screen.get_width()//2 - 5, 0), 5)
        pygame.draw.line(self.screen, self.blue, (0, self.screen.get_height()), (self.screen.get_width()//2 - 5, self.screen.get_height()), 5)
        pygame.draw.line(self.screen, self.red, (self.screen.get_width()//2+5, 0), (self.screen.get_width(), 0), 5)
        pygame.draw.line(self.screen, self.red, (self.screen.get_width()//2 + 5, self.screen.get_height()), (self.screen.get_width(), self.screen.get_height()), 5)
        pygame.draw.line(self.screen, self.blue, (0, 0), (0, self.screen.get_height()//2-goalheight), 5)
        pygame.draw.line(self.screen, self.blue, (0, self.screen.get_height()//2 + goalheight), (0, self.screen.get_height()), 5)
        pygame.draw.line(self.screen, self.red, (self.screen.get_width(), 0), (self.screen.get_width(), self.screen.get_height()//2-goalheight), 5)
        pygame.draw.line(self.screen, self.red, (self.screen.get_width(), self.screen.get_height()//2 + goalheight), (self.screen.get_width(), self.screen.get_height()), 5)
    
    def update_game_state(self, keys):
        """Update all game objects"""
        # Update paddles
        self.paddle1.update(keys, self.screen.get_width(), self.screen.get_height())
        self.paddle2.update(keys, self.screen.get_width(), self.screen.get_height())
        
        # Update disc
        self.disc.update(self.screen.get_width(), self.screen.get_height())
        
        # Check collisions
        self.disc.check_wall_collision(self.screen.get_width(), self.screen.get_height())
        
        # Check scoring
        score_left, score_right = self.disc.check_side_collision(self.screen.get_width(), self.screen.get_height(), self.goal1, self.goal2)
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
    
    def draw_ui(self):
        """Draw user interface elements"""
        self.message_to_screen("Player 1", self.white, -250, -150, "small")
        self.message_to_screen(str(self.score1), self.white, -200, -150, "small")
        self.message_to_screen("Player 2", self.white, -250, 150, "small")
        self.message_to_screen(str(self.score2), self.white, -200, 150, "small")
    
    def run(self):
        """Main game loop"""
        game_exit = False
        
        while not game_exit:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_exit = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.pause()
            
            # Get input
            keys = pygame.key.get_pressed()
            
            # Collect training data
            current_state = self.get_game_state()
            current_input_vector = self.get_neural_network_input_vector()
            
            # Print game state periodically
            if pygame.time.get_ticks() % 1000 < 17:
                print("Current Game State:")
                print(f"Blue Paddle: ({current_state['blue_paddle_x_raw']:.1f}, {current_state['blue_paddle_y_raw']:.1f})")
                print(f"Red Paddle: ({current_state['red_paddle_x_raw']:.1f}, {current_state['red_paddle_y_raw']:.1f})")
                print(f"Disc: ({current_state['disc_x_raw']:.1f}, {current_state['disc_y_raw']:.1f})")
                print(f"Disc Velocity: ({current_state['disc_velocity_x_raw']:.1f}, {current_state['disc_velocity_y_raw']:.1f})")
                print("Goal Area Status:")
                print(f"  Blue paddle in own goal: {current_state['blue_paddle_in_own_goal']}")
                print(f"  Red paddle in own goal: {current_state['red_paddle_in_own_goal']}")
                print(f"Neural Network Input Vector: {[f'{x:.3f}' for x in current_input_vector]}")
                print("-" * 50)
            
            # Record actions
            player1_action = self.paddle1.get_action_vector(keys)
            player2_action = self.paddle2.get_action_vector(keys)
            
            # Store training data
            self.game_states.append(current_input_vector)
            self.player1_actions.append(player1_action)
            self.player2_actions.append(player2_action)
            
            # Update game state
            self.update_game_state(keys)
            
            # Draw everything
            self.draw_field()
            self.draw_ui()
            self.paddle1.draw(self.screen)
            self.paddle2.draw(self.screen)
            self.disc.draw(self.screen)
            
            # Update display
            pygame.display.update()
            self.clock.tick(60)
        
        # Game ended
        print(f"Game ended. Collected {len(self.game_states)} training samples.")
        pygame.quit()
        quit()

# Initialize and run the game
if __name__ == "__main__":
    game = Game()
    game.run()