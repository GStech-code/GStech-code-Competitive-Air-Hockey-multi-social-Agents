# src/environments/game_core/helper_functions.py
"""
Helper functions for the air hockey game.
Includes drawing utilities and game state helpers.
"""

import math
from OpenGL.GL import *


def draw_two_digit_score(game, score, x, y, color):
    """
    Draw a two-digit score using simple rectangles.
    This replaces any complex font rendering with basic shapes.
    """
    # Clamp score to two digits
    score = min(99, max(0, score))
    
    tens = score // 10
    ones = score % 10
    
    digit_width = 15
    digit_height = 25
    digit_spacing = 20
    
    # Draw tens digit
    if tens > 0:
        draw_digit(game, tens, x, y, digit_width, digit_height, color)
    
    # Draw ones digit
    draw_digit(game, ones, x + digit_spacing, y, digit_width, digit_height, color)


def draw_digit(game, digit, x, y, width, height, color):
    """
    Draw a single digit (0-9) using rectangles.
    """
    # Segment dimensions
    h_seg_width = width
    h_seg_height = 3
    v_seg_width = 3
    v_seg_height = height // 2 - 2
    
    # Segment positions
    top = y
    middle = y + height // 2
    bottom = y + height
    left = x
    right = x + width - v_seg_width
    
    # Define which segments are active for each digit
    # Segments: top, top_right, top_left, middle, bottom_right, bottom_left, bottom
    digit_patterns = {
        0: [1, 1, 1, 0, 1, 1, 1],
        1: [0, 1, 0, 0, 1, 0, 0],
        2: [1, 1, 0, 1, 0, 1, 1],
        3: [1, 1, 0, 1, 1, 0, 1],
        4: [0, 1, 1, 1, 1, 0, 0],
        5: [1, 0, 1, 1, 1, 0, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 0, 0, 1, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 1, 0, 1]
    }
    
    pattern = digit_patterns.get(digit, [0, 0, 0, 0, 0, 0, 0])
    
    glColor3f(*color)
    
    # Draw segments based on pattern
    if pattern[0]:  # top
        glBegin(GL_QUADS)
        glVertex2f(left, top)
        glVertex2f(left + h_seg_width, top)
        glVertex2f(left + h_seg_width, top + h_seg_height)
        glVertex2f(left, top + h_seg_height)
        glEnd()
    
    if pattern[1]:  # top_right
        glBegin(GL_QUADS)
        glVertex2f(right, top + h_seg_height)
        glVertex2f(right + v_seg_width, top + h_seg_height)
        glVertex2f(right + v_seg_width, middle)
        glVertex2f(right, middle)
        glEnd()
    
    if pattern[2]:  # top_left
        glBegin(GL_QUADS)
        glVertex2f(left, top + h_seg_height)
        glVertex2f(left + v_seg_width, top + h_seg_height)
        glVertex2f(left + v_seg_width, middle)
        glVertex2f(left, middle)
        glEnd()
    
    if pattern[3]:  # middle
        glBegin(GL_QUADS)
        glVertex2f(left, middle)
        glVertex2f(left + h_seg_width, middle)
        glVertex2f(left + h_seg_width, middle + h_seg_height)
        glVertex2f(left, middle + h_seg_height)
        glEnd()
    
    if pattern[4]:  # bottom_right
        glBegin(GL_QUADS)
        glVertex2f(right, middle + h_seg_height)
        glVertex2f(right + v_seg_width, middle + h_seg_height)
        glVertex2f(right + v_seg_width, bottom)
        glVertex2f(right, bottom)
        glEnd()
    
    if pattern[5]:  # bottom_left
        glBegin(GL_QUADS)
        glVertex2f(left, middle + h_seg_height)
        glVertex2f(left + v_seg_width, middle + h_seg_height)
        glVertex2f(left + v_seg_width, bottom)
        glVertex2f(left, bottom)
        glEnd()
    
    if pattern[6]:  # bottom
        glBegin(GL_QUADS)
        glVertex2f(left, bottom - h_seg_height)
        glVertex2f(left + h_seg_width, bottom - h_seg_height)
        glVertex2f(left + h_seg_width, bottom)
        glVertex2f(left, bottom)
        glEnd()


def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def distance_between_points(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def angle_between_points(x1, y1, x2, y2):
    """Calculate angle from point 1 to point 2"""
    return math.atan2(y2 - y1, x2 - x1)


def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))


def lerp(a, b, t):
    """Linear interpolation between a and b by factor t"""
    return a + (b - a) * t


def vector_magnitude(x, y):
    """Calculate magnitude of a 2D vector"""
    return math.sqrt(x*x + y*y)


def normalize_vector(x, y):
    """Normalize a 2D vector to unit length"""
    mag = vector_magnitude(x, y)
    if mag == 0:
        return 0, 0
    return x / mag, y / mag


def dot_product(x1, y1, x2, y2):
    """Calculate dot product of two 2D vectors"""
    return x1 * x2 + y1 * y2


def reflect_vector(vx, vy, nx, ny):
    """Reflect vector v off a surface with normal n"""
    dot = 2 * dot_product(vx, vy, nx, ny)
    return vx - dot * nx, vy - dot * ny


def point_in_circle(px, py, cx, cy, radius):
    """Check if point is inside circle"""
    return distance_between_points(px, py, cx, cy) <= radius


def circle_circle_collision(x1, y1, r1, x2, y2, r2):
    """Check collision between two circles"""
    return distance_between_points(x1, y1, x2, y2) <= (r1 + r2)


def point_in_rectangle(px, py, rx, ry, rw, rh):
    """Check if point is inside rectangle"""
    return rx <= px <= rx + rw and ry <= py <= ry + rh


# Game state analysis helpers

def calculate_disc_trajectory(disc_x, disc_y, disc_vx, disc_vy, steps_ahead=10):
    """Predict disc position after given number of steps"""
    # Simple linear prediction (ignoring walls and friction for now)
    future_x = disc_x + disc_vx * steps_ahead
    future_y = disc_y + disc_vy * steps_ahead
    return future_x, future_y


def calculate_interception_point(paddle_x, paddle_y, paddle_speed, 
                               disc_x, disc_y, disc_vx, disc_vy):
    """Calculate optimal interception point for paddle to reach disc"""
    # Solve for time when paddle and disc will meet
    # This is a simplified version - real implementation would be more complex
    
    if disc_vx == 0 and disc_vy == 0:
        return disc_x, disc_y
    
    # Try different time intervals to find interception
    best_time = 0
    best_distance = float('inf')
    
    for t in range(1, 100):  # Check up to 100 time steps
        # Where will disc be at time t?
        future_disc_x = disc_x + disc_vx * t
        future_disc_y = disc_y + disc_vy * t
        
        # How far would paddle need to travel?
        travel_distance = distance_between_points(paddle_x, paddle_y, future_disc_x, future_disc_y)
        
        # Can paddle reach in time?
        max_paddle_distance = paddle_speed * t
        
        if travel_distance <= max_paddle_distance:
            if travel_distance < best_distance:
                best_distance = travel_distance
                best_time = t
    
    if best_time > 0:
        intercept_x = disc_x + disc_vx * best_time
        intercept_y = disc_y + disc_vy * best_time
        return intercept_x, intercept_y
    else:
        # Can't intercept, just aim for current position
        return disc_x, disc_y


def calculate_shot_angle(paddle_x, paddle_y, goal_x, goal_y, goal_height):
    """Calculate optimal shooting angle toward goal"""
    # Aim for center of goal
    goal_center_y = goal_y + goal_height / 2
    
    dx = goal_x - paddle_x
    dy = goal_center_y - paddle_y
    
    return math.atan2(dy, dx)


def is_clear_shot(paddle_x, paddle_y, goal_x, goal_y, opponents, obstacle_radius=0.05):
    """Check if there's a clear shot to goal without opponents blocking"""
    
    shot_angle = angle_between_points(paddle_x, paddle_y, goal_x, goal_y)
    shot_distance = distance_between_points(paddle_x, paddle_y, goal_x, goal_y)
    
    # Check if any opponent is blocking the shot
    for opponent in opponents:
        opp_x, opp_y = opponent
        
        # Calculate perpendicular distance from opponent to shot line
        # Using point-to-line distance formula
        dx = goal_x - paddle_x
        dy = goal_y - paddle_y
        
        if dx == 0 and dy == 0:
            continue
        
        # Project opponent position onto shot line
        t = ((opp_x - paddle_x) * dx + (opp_y - paddle_y) * dy) / (dx*dx + dy*dy)
        
        # Clamp t to line segment
        t = clamp(t, 0, 1)
        
        # Find closest point on line
        closest_x = paddle_x + t * dx
        closest_y = paddle_y + t * dy
        
        # Check distance
        distance_to_line = distance_between_points(opp_x, opp_y, closest_x, closest_y)
        
        if distance_to_line < obstacle_radius:
            return False
    
    return True


# Formation analysis helpers

def calculate_team_spread(paddle_positions):
    """Calculate how spread out a team is"""
    if len(paddle_positions) < 2:
        return 0
    
    total_distance = 0
    count = 0
    
    for i in range(len(paddle_positions)):
        for j in range(i + 1, len(paddle_positions)):
            x1, y1 = paddle_positions[i]
            x2, y2 = paddle_positions[j]
            total_distance += distance_between_points(x1, y1, x2, y2)
            count += 1
    
    return total_distance / count if count > 0 else 0


def calculate_formation_center(paddle_positions):
    """Calculate the center point of a team formation"""
    if not paddle_positions:
        return 0, 0
    
    total_x = sum(pos[0] for pos in paddle_positions)
    total_y = sum(pos[1] for pos in paddle_positions)
    
    return total_x / len(paddle_positions), total_y / len(paddle_positions)


def evaluate_defensive_positioning(paddle_positions, own_goal_x, own_goal_y):
    """Evaluate how well positioned a team is for defense"""
    if not paddle_positions:
        return 0
    
    # Calculate average distance to own goal
    total_distance = 0
    for x, y in paddle_positions:
        total_distance += distance_between_points(x, y, own_goal_x, own_goal_y)
    
    avg_distance = total_distance / len(paddle_positions)
    
    # Closer to goal = better defensive positioning
    # Normalize based on field size (assuming 0-1 coordinates)
    max_distance = math.sqrt(2)  # Diagonal of unit square
    
    return 1.0 - (avg_distance / max_distance)


def evaluate_offensive_positioning(paddle_positions, opponent_goal_x, opponent_goal_y):
    """Evaluate how well positioned a team is for offense"""
    if not paddle_positions:
        return 0
    
    # Similar to defensive but inverted - closer to opponent goal is better
    total_distance = 0
    for x, y in paddle_positions:
        total_distance += distance_between_points(x, y, opponent_goal_x, opponent_goal_y)
    
    avg_distance = total_distance / len(paddle_positions)
    max_distance = math.sqrt(2)
    
    return 1.0 - (avg_distance / max_distance)