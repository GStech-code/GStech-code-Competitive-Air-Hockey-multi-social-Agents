import pygame
import math
import numpy as np
from pygame.locals import *


class Paddle:
    def __init__(self, x, y, radius, velocity, side='left', letter = 'A',  player_controlled = True):
        self.center_x = x
        self.center_y = y
        self.radius = radius
        self.current_speed = abs(velocity)
        self.velocity = velocity
        self.side = side  # 'left' or 'right'
        self.letter = letter # 'A' or 'B'
        self.player_controlled = player_controlled # Control flag for Player (True) or NEAT (False)
        self.color = (255, 100, 100) if side == 'left' else (20, 20, 100)
        self.num_in_goal = 0
        self.times_not_moving = 0
        self.touching_wall = False
        
        # Enhanced movement system
        self.acceleration = 0.8
        self.deceleration = 0.0
        self.max_velocity = velocity * 1.5
        self.current_velocity_x = 0
        self.current_velocity_y = 0
        
        # Track previous position for velocity calculation
        self.prev_x = x
        self.prev_y = y
        self.actual_velocity_x = 0
        self.actual_velocity_y = 0
        self.actual_speed = 0
        # Input smoothing
        self.input_history = []
        self.input_history_size = 3
        
        # Enhanced collision detection
        self.collision_buffer = 2  # Prevent paddles from getting too close
    
    def smooth_input(self, input_vector):
        """Smooth input for better responsiveness"""
        self.input_history.append(input_vector)
        if len(self.input_history) > self.input_history_size:
            self.input_history.pop(0)
        
        # Calculate weighted average of recent inputs
        if len(self.input_history) == 0:
            return [0, 0, 0, 0]
        
        weights = [0.5, 0.3, 0.2][:len(self.input_history)]
        smoothed = [0, 0, 0, 0]
        
        for i, weight in enumerate(weights):
            for j in range(4):
                smoothed[j] += self.input_history[-(i+1)][j] * weight
        
        return smoothed
    
    def update(self, keys, screen_width, screen_height):
        """Update paddle position with enhanced responsiveness"""
        # Store previous position
        self.prev_x = self.center_x
        self.prev_y = self.center_y
        
        
        # Get raw input based on whether keys is a dict or ScancodeWrapper
        blueA_down, blueA_up, redA_up, redA_down = 0, 0, 0, 0
        blueB_down, blueB_up, redB_up, redB_down = 0, 0, 0, 0
        blueA_left, blueA_right, redA_right, redA_left = 0, 0, 0, 0
        blueB_left, blueB_right, redB_right, redB_left = 0, 0, 0, 0
        
        if isinstance(keys, dict): # NEAT input
            if self.side == 'left':
                if self.letter == 'A':
                    blueA_left = 1 if keys.get(K_a, False) else 0
                    blueA_right = 1 if keys.get(K_d, False) else 0
                    blueA_up = 1 if keys.get(K_w, False) else 0
                    blueA_down = 1 if keys.get(K_s, False) else 0
                else:
                    blueB_left = 1 if keys.get(K_f, False) else 0
                    blueB_right = 1 if keys.get(K_h, False) else 0
                    blueB_up = 1 if keys.get(K_t, False) else 0
                    blueB_down = 1 if keys.get(K_g, False) else 0
            else: # right paddle
                if self.letter == 'A':
                    red_left = 1 if keys.get(K_LEFT, False) else 0
                    red_right = 1 if keys.get(K_RIGHT, False) else 0
                    red_up = 1 if keys.get(K_UP, False) else 0
                    red_down = 1 if keys.get(K_DOWN, False) else 0
                else:
                    red_left = 1 if keys.get(K_j, False) else 0
                    red_right = 1 if keys.get(K_l, False) else 0
                    red_up = 1 if keys.get(K_i, False) else 0
                    red_down = 1 if keys.get(K_k, False) else 0
        else: # pygame.key.ScancodeWrapper (player input)
            if self.side == 'left':
                if self.letter== 'A': # WASD controls
                    if keys[K_a]:
                        blueA_left = 1
                    if keys[K_d]:
                        blueA_right = 1
                    if keys[K_w]:
                        blueA_up = 1
                    if keys[K_s]:
                        blueA_down = 1
                else: # TFGH controls
                    if keys[K_f]:
                        blueB_left = 1
                    if keys[K_h]:
                        blueB_right = 1
                    if keys[K_t]:
                        blueB_up = 1
                    if keys[K_g]:
                        blueB_down = 1
            else:
                if self.letter== 'A': # Arrow key controls
                    if keys[K_LEFT]:
                        redA_left = 1
                    if keys[K_RIGHT]:
                        redA_right = 1
                    if keys[K_UP]:
                        redA_up = 1
                    if keys[K_DOWN]:
                        redA_down = 1
                else: # IJKL controls
                    if keys[K_j]:
                        redB_left = 1
                    if keys[K_l]:
                        redB_right = 1
                    if keys[K_i]:
                        redB_up = 1
                    if keys[K_k]:
                        redB_down = 1
        
        if self.side == 'left':
            if self.letter== 'A':
                raw_input = [blueA_left, blueA_right, blueA_up, blueA_down]
            else:
                raw_input = [blueB_left, blueB_right, blueB_up, blueB_down]
        else:
            if self.letter== 'A':
                raw_input = [redA_left, redA_right, redA_up, redA_down]
            else:
                raw_input = [redB_left, redB_right, redB_up, redB_down]
        
        # Apply input smoothing
        smoothed_input = self.smooth_input(raw_input)
        
        # Calculate desired movement
        if self.side == 'left':
            desired_x = (smoothed_input[1] - smoothed_input[0])  # right - left
            desired_y = (smoothed_input[3] - smoothed_input[2])  # down - up
        else:
            desired_x = (smoothed_input[1] - smoothed_input[0])  # right - left
            desired_y = (smoothed_input[3] - smoothed_input[2])  # down - up
        
        # Apply acceleration/deceleration
        if desired_x != 0:
            self.current_velocity_x += desired_x * self.acceleration
        else:
            self.current_velocity_x *= self.deceleration
        
        if desired_y != 0:
            self.current_velocity_y += desired_y * self.acceleration
        else:
            self.current_velocity_y *= self.deceleration
        
        # Clamp velocities
        self.current_velocity_x = max(-self.max_velocity, min(self.max_velocity, self.current_velocity_x))
        self.current_velocity_y = max(-self.max_velocity, min(self.max_velocity, self.current_velocity_y))
        
        # Apply movement
        self.center_x += self.current_velocity_x
        self.center_y += self.current_velocity_y
        
        # Apply boundaries
        self.apply_boundaries(screen_width, screen_height)
        
        # Calculate actual velocity (how much the paddle actually moved)
        self.actual_velocity_x = self.center_x - self.prev_x
        self.actual_velocity_y = self.center_y - self.prev_y
        actual_speed = math.sqrt(self.actual_velocity_x**2 + self.actual_velocity_y**2)
        self.actual_speed = actual_speed if actual_speed >= 0.05 else 0
        self.times_not_moving += 1 if self.actual_speed <= 0.05 else 0

    
    def apply_boundaries(self, screen_width, screen_height):
        """Apply movement boundaries with collision buffer"""
        # Top and bottom boundaries
        if self.center_y - self.radius < self.collision_buffer:
            self.center_y = self.radius + self.collision_buffer
            self.current_velocity_y = 0
            #self.touching_wall = True

        elif self.center_y + self.radius > screen_height - self.collision_buffer:
            self.center_y = screen_height - self.radius - self.collision_buffer
            self.current_velocity_y = 0
            #self.touching_wall = True
        
        # Side boundaries
        if self.side == 'left':
            # Left paddle boundaries
            if self.center_x - self.radius < self.collision_buffer:
                self.center_x = self.radius + self.collision_buffer
                self.current_velocity_x = 0
                self.touching_wall = True
            elif self.center_x + self.radius > screen_width / 2 - self.collision_buffer:
                self.center_x = screen_width / 2 - self.radius - self.collision_buffer
                self.current_velocity_x = 0
                self.touching_wall = True
            
            else:
                self.touching_wall = False
        
        else:
            # Right paddle boundaries
            if self.center_x + self.radius > screen_width - self.collision_buffer:
                self.center_x = screen_width - self.radius - self.collision_buffer
                self.current_velocity_x = 0
                self.touching_wall = True

            elif self.center_x - self.radius < screen_width / 2 + self.collision_buffer:
                self.center_x = screen_width / 2 + self.radius + self.collision_buffer
                self.current_velocity_x = 0
                self.touching_wall = True
            
            else:
                self.touching_wall = False

    
    
    def is_in_goal_area(self, screen_width, screen_height):
        """Check if paddle is in goal area"""
        goal_circle_radius = screen_width / 10
        
        if self.side == 'left':
            goal_center = (0, screen_height / 2)
            distance = math.sqrt((self.center_x - goal_center[0])**2 + (self.center_y - goal_center[1])**2)
            if distance <= goal_circle_radius and self.center_x >= 0:
                self.num_in_goal += 1
            return distance <= goal_circle_radius and self.center_x >= 0
        else:
            goal_center = (screen_width, screen_height / 2)
            distance = math.sqrt((self.center_x - goal_center[0])**2 + (self.center_y - goal_center[1])**2)
            if distance <= goal_circle_radius and self.center_x <= screen_width:
                self.num_in_goal += 1
            return distance <= goal_circle_radius and self.center_x <= screen_width
    


    def check_collision_with_other_paddle(self, other):
        dx = other.center_x - self.center_x
        dy = other.center_y - self.center_y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        min_distance = self.radius + other.radius + self.collision_buffer

        if distance < min_distance and distance != 0:
            overlap = min_distance - distance
            push_x = (dx / distance) * (overlap / 2)
            push_y = (dy / distance) * (overlap / 2)

            self.center_x -= push_x
            self.center_y -= push_y
            other.center_x += push_x
            other.center_y += push_y

            self.current_velocity_x *= 0.5
            self.current_velocity_y *= 0.5
            other.current_velocity_x *= 0.5
            other.current_velocity_y *= 0.5

    
    def reset_position(self, x, y):
        """Reset paddle to specific position"""
        self.center_x = x
        self.center_y = y
        self.current_velocity_x = 0
        self.current_velocity_y = 0
        self.actual_velocity_x = 0
        self.actual_velocity_y = 0
        self.actual_speed = 0
        self.input_history = []
        self.num_in_goal = 0
        self.times_not_moving = 0
        self.touching_wall = False