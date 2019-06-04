
import neat
from neat.six_util import iteritems, itervalues
import pickle

selected_genomes = [941, 929, 871, 99, 942, 365, 245, 1042, 594, 785, 927,
                    31, 829, 10, 18, 17, 21, 22, 26, 32, 36, 40, 54, 84, 98,
                    103, 106, 112, 122, 152, 164, 176, 182, 174, 234, 227,
                    261, 262, 266, 307, 310, 315, 317, 314, 318, 332, 336,
                    356, 372, 383, 385, 387, 392, 403, 400]

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
