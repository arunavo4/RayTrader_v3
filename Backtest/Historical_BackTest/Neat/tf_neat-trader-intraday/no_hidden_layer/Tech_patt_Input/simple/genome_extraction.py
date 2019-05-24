
import neat
from neat.six_util import iteritems, itervalues
import pickle


restore_file = "neat-checkpoint-50"
pop = neat.Checkpointer.restore_checkpoint(restore_file)


for g in itervalues(pop.population):
    print(str(g.key))
    save_path = './genomes/' + str(g.key) + '.pkl'
    with open(save_path, 'wb') as output:
        pickle.dump(g, output, 1)