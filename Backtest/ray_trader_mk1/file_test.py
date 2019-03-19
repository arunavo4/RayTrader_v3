import glob
import pickle


# Load the tick data and call on_tick()
print("Loading tick data")
data_dir = [ ".\\live_ticks\\2019-03-06\\" + str(i) + ".out" for i in range(1,400)]
for in_file in data_dir:
    print("Processing :", in_file)
    # unpickle each file and the feed to on_ticks()
    # live_ticks = []
    # # Read from the file the stock list
    # with open(in_file, 'rb') as fp:
    #     live_ticks = pickle.load(fp)
    #
    # for ticks in live_ticks:
    #     print(ticks)
    #     break
    #
    # break
