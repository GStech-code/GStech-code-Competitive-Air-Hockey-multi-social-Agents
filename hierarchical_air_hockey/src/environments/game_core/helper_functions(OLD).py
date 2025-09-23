def draw_two_digit_score(self, score, x, y, color):
    """Draw a 2-digit score (00-99) using 7-segment display style"""
    # Ensure score is within 0-99 range
    score = max(0, min(99, score))
    
    # Get tens and units digits
    tens = score // 10
    units = score % 10
    
    # Draw tens digit
    draw_simple_digit(self, tens, x, y, color)
    
    # Draw units digit (offset by digit width + spacing)
    draw_simple_digit(self, units, x + 20, y, color)

def draw_simple_digit(self, digit, x, y, color):
    """Draw a simple digit using line segments (7-segment display style)"""
    if digit < 0 or digit > 9:
        return
        
    # 7-segment display patterns
    patterns = {
        0: [1, 1, 1, 1, 1, 1, 0],  # top, top-right, bottom-right, bottom, bottom-left, top-left, middle
        1: [0, 1, 1, 0, 0, 0, 0],
        2: [1, 1, 0, 1, 1, 0, 1],
        3: [1, 1, 1, 1, 0, 0, 1],
        4: [0, 1, 1, 0, 0, 1, 1],
        5: [1, 0, 1, 1, 0, 1, 1],
        6: [1, 0, 1, 1, 1, 1, 1],
        7: [1, 1, 1, 0, 0, 0, 0],
        8: [1, 1, 1, 1, 1, 1, 1],
        9: [1, 1, 1, 1, 0, 1, 1]
    }
    
    pattern = patterns.get(digit, [0, 0, 0, 0, 0, 0, 0])
    segment_length = 15
    
    # Draw segments based on pattern
    if pattern[0]:  # top
        self.draw_line(x, y, x + segment_length, y, color, 2)
    if pattern[1]:  # top-right
        self.draw_line(x + segment_length, y, x + segment_length, y + segment_length, color, 2)
    if pattern[2]:  # bottom-right
        self.draw_line(x + segment_length, y + segment_length, x + segment_length, y + 2 * segment_length, color, 2)
    if pattern[3]:  # bottom
        self.draw_line(x, y + 2 * segment_length, x + segment_length, y + 2 * segment_length, color, 2)
    if pattern[4]:  # bottom-left
        self.draw_line(x, y + segment_length, x, y + 2 * segment_length, color, 2)
    if pattern[5]:  # top-left
        self.draw_line(x, y, x, y + segment_length, color, 2)
    if pattern[6]:  # middle
        self.draw_line(x, y + segment_length, x + segment_length, y + segment_length, color, 2)