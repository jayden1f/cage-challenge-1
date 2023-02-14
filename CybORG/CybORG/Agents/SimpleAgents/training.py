import subprocess
import time
import os
import sys

run_string = 'Start'
runs = 1
input_start = 0
input_end = 0

cur_string = input("Enter verification code: \n")
while cur_string == run_string:
    os.system('clear')
    print("Commencing Training")
    print("Training Round: ", runs)
    print("Previous Time: ", input_end - input_start, "sec")
    #stdoutOrigin=sys.stdout
    #sys.stdout = open("result.txt","a")
    #sys.stdout.close()
    input_start = time.time()
    subprocess.call('python SkyNet.py', shell=True)
    #time.sleep(300)
    input_end = time.time()
    out_string = "Round: " + str(runs) + "   Previous Time: " + str(input_end - input_start)+ " sec\n"
    f = open("result.txt", "a")
    f.write(out_string)
    f.close()
    runs += 1
    #cur_string = input("Enter verification code: \n")
    #time.sleep(5)

print("Closing training simulation")