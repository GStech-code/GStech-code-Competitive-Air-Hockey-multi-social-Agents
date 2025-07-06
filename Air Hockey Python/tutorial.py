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


def eval_genomes(genomes, config):
    for i, (genome_id1, genome1) in enumerate(genomes):
        for genome_id2, genome2 in genomes[i+1:]:
            pass


def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-27')
    p = neat.population(config) # coment out this line if loading from checkpoint!!!
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