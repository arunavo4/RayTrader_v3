
import neat
from neat.six_util import iteritems, itervalues
import pickle


selected_genomes = [941, 871, 942, 99, 1072, 245, 785, 937, 927, 31, 4, 2, 1,
                    18, 19, 23, 31, 37, 42, 46, 52, 50, 65, 69, 74, 76, 79, 84,
                    94, 98, 100, 108, 105, 118, 120, 124, 129, 132, 140, 144, 143,
                    146, 147, 151, 152, 159, 160, 153, 171, 169, 176, 178, 179, 180,
                    187, 192, 199, 216, 207, 218, 220, 222, 251, 257, 268, 279, 308,
                    310, 332, 334, 345, 349, 353, 358, 366, 376, 385, 387, 401]

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
