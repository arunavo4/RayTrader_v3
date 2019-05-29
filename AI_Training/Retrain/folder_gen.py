import glob
import os

files = glob.glob("G:\\AI Trading\\Code\\RayTrader_v3\\HistoricalData\\Min_data\\*.csv")

for f in files:
    dir_name = os.path.split(f)[-1]
    # Create a dir to store today's order data
    dir_path = ".\\" + str(dir_name.split(".")[0]) + "\\"

    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)  # succeeds even if directory exists.
        except FileExistsError:
            print("Dir creation failed!")
            pass

