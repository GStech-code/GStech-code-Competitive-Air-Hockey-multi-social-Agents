import pygame
import math
import random
from pygame.locals import *


class Disc:
    def __init__(self, x, y, radius):
        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.max_speed = 6
        angle = self._get_random_angle(-30, 30, [0])
        self.x_vel = abs(math.cos(angle) * self.max_speed)
        self.y_vel = math.sin(angle) * self.max_speed

        #self.image = pygame.image.load(image_path)
        #self.image = pygame.transform.scale(self.image, (radius * 2, radius * 2))
    
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
        restitution = 0.75  # Slightly less than perfectly elastic for realistic feel
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