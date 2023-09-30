import math

def sig(x):
    #use logistic function as activation function
    return float(1) / float(1 + math.exp(-x))

def inv_sig(x):
    #derivative of the output of neruon with respect to its input
    return sig(x) * (1 - sig(x))

def err(o, t):
    #squared error function, o is the actual output value and t is the target output 
    return 0.5 * ((t - o) ** 2)

def inv_err(o, t):
    #derivative of squared error function with respect to o
    return (o - t)


