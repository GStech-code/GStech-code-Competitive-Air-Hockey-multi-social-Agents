from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT
from air_hockey_python import Game
import multiprocessing as mp
from tqdm import tqdm
import pickle
import pygame
import neat
import time
import os


class HockeyGame:
    def __init__(self):
        self.game = Game(render= False)
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
            
            if (game_info['score1'] >= 5 and game_info['num_blue_hits'] > 0) or (self.game.score2 >= 5 and game_info['num_red_hits'] > 0) or game_info['game_time'] > 20:
                #if game_info['game_time'] > 20:
                #    print("\nTime out!")
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
        blue_fitness -= red_score / 2
        red_fitness -= blue_score / 2
        
        # Penalty for being in own goal area (defensive behavior)
        blue_fitness -= game_info['blue_paddle_num_in_goal'] * game_info['blue_paddle_in_own_goal']
        red_fitness -= game_info['red_paddle_num_in_goal'] * game_info['red_paddle_in_own_goal']
        
        # Reward active play - to hit the disk
        blue_fitness += 0.25 * game_info['num_blue_hits']
        red_fitness += 0.25 * game_info['num_red_hits']

        # Penalty if the other player hits the disk
        blue_fitness -= 0.25 * game_info['num_red_hits']
        red_fitness -= 0.25 * game_info['num_blue_hits']
        
        
        # Ensure minimum fitness (avoid negative values that could cause issues)
        blue_fitness = max(0, blue_fitness)
        red_fitness = max(0, red_fitness)
        
        # Assign fitness to genomes
        genome1.fitness += blue_fitness
        genome2.fitness += red_fitness
        
        # Optional: Print fitness for debugging
        #print(f"Blue fitness: {blue_fitness:.2f}, Red fitness: {red_fitness:.2f}")
        #print(f"Game ended - Blue: {blue_score}, Red: {red_score}")

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
    
    # Aggregate fitness results back to original genomes
    for genome1_id, genome2_id, fitness1, fitness2 in results:
        genome_dict[genome1_id].fitness += fitness1
        genome_dict[genome2_id].fitness += fitness2
    
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
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-6')
    #p = neat.Population(config)
    
    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    p.add_reporter(ProgressReporter(max_generations))
    
    # Run evolution with parallel evaluation
    print(f"Starting NEAT training with {max_generations} generations...")
    print(f"Using {min(mp.cpu_count(), 8)} CPU cores for parallel processing\n")
    
    winner = p.run(eval_genomes_parallel, max_generations)
    
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
    run_neat(config)
    #test_ai(config)