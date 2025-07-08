import neat.population
import pygame
from air_hockey_python import Game
import neat
import os


class HockeyGame:
    def __init__(self):
        self.game = Game()
        self.paddle1 = self.game.paddle1
        self.paddle2 = self.game.paddle2
        self.disc = self.game.disc


    def test_ai(self):
        self.game.run()


    def train_ai(self, genome1, genome2, config):
        render_debug = True  # Set to False for full-speed NEAT training
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
            
            print(f"Blue decision: {decision1}\nRed decision: {decision2}")

            # Just advance one frame of game logic, no OpenGL rendering
            game_info = self.game.update_one_frame(decision1, decision2, render= render_debug)
            
            if self.game.score1 >= 1 or self.game.score2 >= 1:
                print("game_info = ", game_info)
                self.calculate_fitness(genome1= genome1, genome2= genome2, game_info= game_info)
                break
        
        pygame.quit()


    def calculate_fitness(self, genome1, genome2, game_info):
    # Calculate rewards (you can customize this)
        blue_reward = 0
        red_reward = 0
        
        # Score rewards
        if new_state['score1'] > prev_state['score1']:
            blue_reward += 100  # Blue scored
            red_reward -= 100   # Red got scored on
        if new_state['score2'] > prev_state['score2']:
            red_reward += 100   # Red scored  
            blue_reward -= 100  # Blue got scored on


def eval_genomes(genomes, config):
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) -1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            hockey_game = HockeyGame()
            hockey_game.train_ai(genome1, genome2, config)


def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-27')
    p = neat.Population(config) # coment out this line if loading from checkpoint!!!
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    winner = p.run(eval_genomes, 50)



if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
    