import random
import math

def sigmoid(z):
    if z>700:
        return 1.0
    elif z<-700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))

if __name__ == '__main__':
    print(sigmoid(50))