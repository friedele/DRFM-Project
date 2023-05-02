import matplotlib.pyplot as plt

training_acc=[10,20,30,40,45,50,55,60,65,70,80,85,86,89,90,92,93,94,95,98,99]
testing_acc=[5,15,25,30,40,45,50,60,61,65,70,75,76,79,80,82,83,84,85,88,90]

##### Plots ##### 
plt.plot(training_acc)
plt.plot(testing_acc)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()