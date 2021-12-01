#Ipek Erdogan
#2019700174
#Pattern Recognition HW3
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def data_loader(path):
    data = np.load(path)
    return data
def testing(test_features,test_labels,w1,w2,b):
    preds=[]
    test_losses=[]
    for i in range(len(test_features)):
        x = test_features[i]
        y = test_labels[i]
        Z = w1 * x[0] + w2 * x[1] + b
        h = math.tanh(Z)
        if (h >= 0):
            pred = 1
        else:
            pred = -1
        preds.insert(i,pred)
        loss = np.log(1 + np.exp(-y * Z))
        test_losses.append(loss)
    epoch_test_loss = sum(test_losses) / len(test_losses)
    test_accuracy = sum(1 for x, y in zip(preds, test_labels) if x == y) / float(len(preds))
    return test_accuracy,epoch_test_loss

if __name__ == '__main__':
    path1 = 'train_features.npy'
    train_features = data_loader(path1)
    path2 = 'train_labels.npy'
    train_labels = data_loader(path2)
    path3 = 'test_features.npy'
    test_features = data_loader(path3)
    path4 = 'test_labels.npy'
    test_labels = data_loader(path4)
    w1 = np.random.uniform(0,1)
    w2 = np.random.uniform(0,1)
    b = np.random.uniform(0,1)
    learning_rate=0.001
    train_acc=[]
    test_acc=[]
    train_loss=[]
    test_loss=[]
    epoch=1000
    for k in range(epoch):
        predictions = []
        losses=[]
        indices = np.random.permutation(len(train_features)) #Randomize the indices of the data
        for i in range(len(indices)):
            x=train_features[indices[i]]
            y=train_labels[indices[i]]
            Z = w1*x[0]+w2*x[1]+b
            h = math.tanh(Z)
            if(h>=0): #making the prediction according to the output of the tangent function
                prediction=1
            else:
                prediction=-1
            predictions.insert(indices[i],prediction)
            loss=np.log(1+np.exp(-y*Z)) #Our loss function.
            losses.append(loss)
            #dh/dZ = 1+pow(h,2)
            #dZ/dW1 = x[0]
            #dZ/dW2 = x[1]
            #dZ/b=1
            #chain rule = dloss/dh * dh/dZ*dZ/dW1
            gradient_w1 = (-y * x[0])*(1+pow(h,2)) / (np.exp(y * Z) + 1)
            gradient_w2 = (-y * x[1])*(1+pow(h,2)) / (np.exp(y * Z) + 1)
            gradient_b = (-y)*(1+pow(h,2)) / (np.exp(y * Z) + 1)
            w1 = w1 - learning_rate * gradient_w1
            w2 = w2 - learning_rate * gradient_w2
            b = b - learning_rate * gradient_b

        print("EPOCH :" ,k)
        epoch_train_loss = sum(losses) / len(losses)
        train_accuracy=sum(1 for x,y in zip(predictions,train_labels) if x == y) / float(len(predictions))
        print("Training Accuracy :", train_accuracy)
        print("Epoch Training Loss:", epoch_train_loss)
        test_accuracy,epoch_test_loss=testing(test_features, test_labels, w1, w2, b)
        print("Test Accuracy :", test_accuracy)
        print("Epoch Test Loss:", epoch_test_loss)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        train_loss.append(epoch_train_loss)
        test_loss.append(epoch_test_loss)

    plt.plot(train_acc)
    plt.title('Plot of overall train accuracy to epoch for SGD optimizer')
    plt.ylabel('Train Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('train_acc.png')
    plt.show()
    plt.plot(test_acc)
    plt.title('Plot of overall test accuracy to epoch for SGD optimizer')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('test_acc.png')
    plt.show()
    plt.plot(train_loss)
    plt.title('Plot of overall train loss to epoch for SGD optimizer')
    plt.ylabel('Train Loss')
    plt.xlabel('Epoch')
    plt.savefig('train_loss.png')
    plt.show()
    plt.plot(test_loss)
    plt.title('Plot of overall test loss to epoch for SGD optimizer')
    plt.ylabel('Test Loss')
    plt.xlabel('Epoch')
    plt.savefig('test_loss.png')
    plt.show()
