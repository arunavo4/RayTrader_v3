import glob
import time
import os

# Load the parent dir
# files = glob.glob("/home/ubuntu/zerodha/live_data/" + str(date.today()) + "/*.csv")
files = glob.glob("G:\\AI Trading\\Code\\RayTrader_v3\\Backtest\\Live_BackTest\\live_ticks\\*\\ADANIPORTS.csv")

# threads_dict = {}

#Single Threaded
for f in files:
    print("Processing: ", os.path.split(f)[-1])
    # start = time.time()
    # start_trade(f, os.path.split(f)[-1])
    # end = time.time()
    # print("Stock: {0} Time req: {1}".format(os.path.split(f)[-1],(end-start)))

