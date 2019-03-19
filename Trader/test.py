import threading
import time
from datetime import datetime,date

startup = False

def abc(timestamp):
    global startup
    print("func abc called! :",timestamp)
    time.sleep(5)
    startup = True
    print("after sleeping 5 sec: ",timestamp)


#Create all threads
print("Multi-threaded Program")
t = threading.Thread(target=abc, args=(datetime.now(),))

t.daemon = True
#start all threads
t.start()

#Join all threads
# t.join()

print("Main thread ended")
print("Startup:",startup)

time.sleep(7)

print("Startup:",startup)
