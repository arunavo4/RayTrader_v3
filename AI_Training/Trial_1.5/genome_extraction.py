
import neat
from neat.six_util import iteritems, itervalues
import pickle


restore_file = "neat-checkpoint-99"
pop = neat.Checkpointer.restore_checkpoint(restore_file)

selected_genomes = [939, 927]

for g in itervalues(pop.population):
    print(str(g.key))
    if g.key in selected_genomes:
        print(str(g.key))

        save_path = './genomes/' + str(g.key) + '.pkl'
        with open(save_path, 'wb') as output:
            pickle.dump(g, output, 1)