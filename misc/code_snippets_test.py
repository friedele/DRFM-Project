import datetime as dt

path = "C:/Users/friedele/Repos/DRFM/ouputs/" 
date_string  = dt.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
acc_file = date_string + "_accuracy"
val_acc_file= date_string + "_validation_accuracy"