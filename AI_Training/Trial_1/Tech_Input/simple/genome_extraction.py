
import neat
from neat.six_util import iteritems, itervalues
import pickle


restore_file = "neat-checkpoint-21"
pop = neat.Checkpointer.restore_checkpoint(restore_file)

selected_genomes = [110, 25, 109, 32, 65, 158]

for g in itervalues(pop.population):
    if g.key in selected_genomes:
        print(str(g.key))

        save_path = './genomes/' + str(g.key) + '.pkl'
        with open(save_path, 'wb') as output:
            pickle.dump(g, output, 1)