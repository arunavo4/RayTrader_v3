
import neat
from neat.six_util import iteritems, itervalues
import pickle


selected_genomes = [929, 942, 99, 785, 19, 20, 33, 36, 38, 46,
                    84, 140, 160, 186, 194, 206, 259, 267]

print("Total len: ",len(selected_genomes))

for i in range(30):
    restore_file = "neat-checkpoint-" + str(i)
    pop = neat.Checkpointer.restore_checkpoint(restore_file)

    for g in itervalues(pop.population):
        # print(str(g.key))
        if g.key in selected_genomes:
            print(str(g.key))

            save_path = './genomes/' + str(g.key) + '.pkl'
            with open(save_path, 'wb') as output:
                pickle.dump(g, output, 1)
