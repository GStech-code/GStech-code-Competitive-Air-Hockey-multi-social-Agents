from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT
from air_hockey_python import Game
import pickle
import pygame
import neat
import os


class HockeyGame:
    def __init__(self):
        self.game = Game()
        self.paddle1 = self.game.paddle1
        self.paddle2 = self.game.paddle2
        self.disc = self.game.disc


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
            ai_input = (
                self.paddle1.center_x / (self.game.screen_width / 2) * -1,  # Blue paddle x (normalized)
                self.paddle1.center_y / self.game.screen_height,            # Blue paddle y (normalized)
                self.paddle1.current_speed,                                 # Blue paddle speed
                self.paddle2.center_x - (self.game.screen_width / 2),       # Red paddle x (offset)
                self.paddle2.center_y,                                      # Red paddle y
                self.disc.center_x - (self.game.screen_width / 2),          # Disc x (offset)
                self.paddle2.current_speed,                                 # Red paddle speed
                self.disc.center_y,                                         # Disc y
                self.disc.x_vel,                                            # Disc x velocity
                self.disc.y_vel                                             # Disc y velocity
            )
            
            output = net.activate(ai_input)
            ai_decision = self.game.neat_requests(output, 'left')
            
            # Update AI paddle
            self.paddle1.update(ai_decision, self.game.screen_width, self.game.screen_height)
            
            # Human controls red paddle (right side)
            keys = pygame.key.get_pressed()
            self.paddle2.update(keys, self.game.screen_width, self.game.screen_height)
            
            # Update disc physics
            self.disc.update(self.game.screen_width, self.game.screen_height)
            self.disc.check_wall_collision(self.game.screen_width, self.game.screen_height)
            
            # Check scoring
            score_left, score_right = self.disc.check_side_collision(
                self.game.screen_width, self.game.screen_height, self.game.goal1, self.game.goal2
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
        while run:
            output1 = net1.activate((self.paddle1.center_x / (self.game.screen_width / 2) * -1, self.paddle1.center_y / self.game.screen_height, self.paddle1.current_speed, self.paddle2.center_x - (self.game.screen_width / 2), self.paddle2.center_y, self.disc.center_x - (self.game.screen_width / 2), self.paddle2.current_speed, self.disc.center_y, self.disc.x_vel, self.disc.y_vel)) # Blue
            decision1 = self.game.neat_requests(output1, 'left')
            if all(value <= 0.5 for value in decision1.values()):
                pass
            else:
                self.paddle1.update(decision1, self.game.screen_width, self.game.screen_height)
            
            output2 = net2.activate((self.paddle2.center_x - (self.game.screen_width / 2), self.paddle2.center_y, self.paddle2.current_speed, self.paddle1.center_x / (self.game.screen_width / 2) * -1,  self.paddle1.center_y / self.game.screen_height, self.paddle1.current_speed, self.disc.center_x - (self.game.screen_width / 2), self.disc.center_y, self.disc.x_vel, self.disc.y_vel)) # Red
            decision2 = self.game.neat_requests(output2, 'right')
            if all(value <= 0.5 for value in decision2.values()):
                pass
            else:
                self.paddle2.update(decision2, self.game.screen_width, self.game.screen_height)
            
            #print(f"Blue decision: {decision1}\nRed decision: {decision2}")

            # Just advance one frame of game logic, no OpenGL rendering
            game_info = self.game.update_one_frame(decision1, decision2, render= render_debug)
            
            if (game_info['score1'] >= 8 and game_info['num_hits'] > 0) or (self.game.score2 >= 8 and game_info['num_hits'] > 0) or game_info['num_hits'] >= 30 or game_info['game_time'] > 80:
                if game_info['game_time'] > 80:
                    print("Time out!")
                if game_info['num_hits'] >= 32:
                    print(f"passed hits 32 hits")
                #else:
                    #print(f"score1 : score2\n     {game_info['score1']} : {game_info['score2']}")
                self.calculate_fitness(genome1= genome1, genome2= genome2, game_info= game_info)
                break
        
        pygame.quit()


    def calculate_fitness(self, genome1, genome2, game_info):
        """
        Calculate fitness for both genomes based on game performance
        
        Args:
            genome1: Blue paddle genome (left side)
            genome2: Red paddle genome (right side)  
        """
        
        # Initialize fitness values
        blue_fitness = 0
        red_fitness = 0
        
        # Basic scoring rewards/penalties
        blue_score = self.game.score1
        red_score = self.game.score2
        
        # Major rewards for scoring
        blue_fitness += blue_score
        red_fitness += red_score
        
        # Penalties for being scored on
        blue_fitness -= red_score * 0.5
        red_fitness -= blue_score * 0.5
        
        # Penalty for being in own goal area (defensive behavior)
        if game_info['blue_paddle_in_own_goal']:
            #print("Blue in own goal")
            blue_fitness -= 0.1
        if game_info['red_paddle_in_own_goal']:
            #print("Red in own goal")
            red_fitness -= 0.1
        
        # Reward active play - to hit the disk
        if self.disc.check_paddle_collision(self.paddle1) and game_info['blue_paddle_in_own_goal'] == False:
            blue_fitness += 0.1
        if self.disc.check_paddle_collision(self.paddle2) and game_info['red_paddle_in_own_goal'] == False:
            red_fitness += 0.1
        
        
        # Ensure minimum fitness (avoid negative values that could cause issues)
        blue_fitness = max(0, blue_fitness)
        red_fitness = max(0, red_fitness)
        
        # Assign fitness to genomes
        genome1.fitness += blue_fitness
        genome2.fitness += red_fitness
        
        # Optional: Print fitness for debugging
        #print(f"Blue fitness: {blue_fitness:.2f}, Red fitness: {red_fitness:.2f}")
        #print(f"Game ended - Blue: {blue_score}, Red: {red_score}")

def eval_genomes(genomes, config):
    n = 1
    for i, (genome_id1, genome1) in enumerate(genomes):
        #if i == len(genomes) -1:
        #    break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            hockey_game = HockeyGame()
            print(f"\nMatch #{n}")
            hockey_game.train_ai(genome1, genome2, config, render= True)
            n += 1


def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-27')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
        print("Saving the best genome to the pickle file \"best.pickle\"")
        print("\"best.pickle\" has a fitness of", winner.fitness)


def test_ai(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
        hockey_game = HockeyGame()
        hockey_game.test_ai(winner, config)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
    #test_ai(config)