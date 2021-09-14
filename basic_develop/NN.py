# Program to add two matrices using list comprehension
import numpy as np
import math
# def sigmod(x):
#     return 1/(1 + math.exp(-1*x))


# def calcerror(f):
    
#     return (0 - f)
# def gradiant_error(layer,e):
#     return layer * ( 1 -layer ) * e

# X1 =np.array([
#     [1],
#     [2],
#     [0.8]
# ])
# X2=np.array([
#     [1],
#     [2],
#     [-0.1]
#     ])
# layer3=[[5]]


# W1=np.array([[0.5,0.4,-1]])
# W2=np.array([[0.9,1,-1]])
# W3=np.array([[-1.2,1.1,-1]])
# Y3=sigmod(np.dot(W1,X1))
# print("y3",Y3)
# Y4=sigmod(np.dot(W2,X2)) 
# print('Y4',Y4)
# Y5=np.array([
#          [Y3],
#          [Y4],
#          [0.3]]
#          )
# Y5=np.dot(W3,Y5)
# Y5=sigmod(Y5[0])
# print('Y5',Y5)
# e=calcerror(Y5)
# print(e)

# print("calculate the error gradient:",gradiant_error(Y5,e))



class NN:
    gr5=0
    gr3=0
    gr4=0
    def sigmod(self,x):
        
        return 1/(1 + math.exp(-1*x))
    def calcerror(self,f):
        return (0 - f)
    def gradiant_error(self,layer,e):
        return layer * ( 1 -layer ) * e

    def __init__(self,x1,x2,w13,w23,w14,w24,w35,w45,theta3,theta4,theta5):
        self.x1=x1
        self.x2=x2
        self.w13=w13
        self.w14=w14
        self.w23=w23
        self.w24=w24
        self.w35=w35
        self.w45=w45
        self.theta3=theta3
        self.theta4=theta4
        self.theta5=theta5
    def initiateLayer(self):
        X1 =np.array([
            [self.x1],
            [self.x2],
            [self.theta3]
        ])
        X2=np.array([
            [self.x1],
            [self.x2],
            [self.theta4]
            ])
        W1=np.array([[self.w13,self.w23,-1]])
        W2=np.array([[self.w14,self.w24,-1]])
        W3=np.array([[self.w35,self.w45,-1]])
        Y3=self.sigmod(np.dot(W1,X1))
    
        Y4=self.sigmod(np.dot(W2,X2))
        Y5=np.array([ [Y3],[Y4],[self.theta5] ])
        Y5=self.sigmod(Y5[0])
     
        e=self.calcerror(Y5)
        gr=self.gradiant_error(Y5,e)
        return Y3,Y4,Y5,e,gr
    def gradientDesnt(self):
        for i in range(0,5):
            a=0.1
            print(self.x1,
            self.x2,
            self.w13,
            self.w14,
            self.w23,
            self.w24,
            self.w35,
            self.w45,
            self.theta3,
            self.theta4,
            self.theta5)
            Y3,Y4,Y5,e,self.gr5=self.initiateLayer()
            print(Y3,Y4,Y5,e,self.gr5)
            New_w35= a  * Y3 * self.gr5
            New_w45= a * Y4 * self.gr5
            New_theta5 = a *(-1) * self.gr5
            self.w35=New_w35+self.w35
            self.w45=New_w45+self.w45
            self.theta5=New_theta5+self.theta5


            self.gr3= Y3 * (1- Y3) * self.gr5 * self.w35
            self.gr4= Y4 * (1 - Y4) * self.gr5 * self.w45

            New_w13= a  * self.x1 * self.gr3
            New_w14= a * self.x1 * self.gr3
            New_theta3 = a * (-1) * self.gr3

            self.w13=New_w13 + self.w13
            self.w14=New_w14 + self.w14
            self.theta3=New_theta3 + self.theta3

            New_w23= a  * self.x2 * self.gr4
            New_w24= a * self.x2 * self.gr4
            New_theta4 = a *(-1) * self.gr4
            self.w23=New_w23 + self.w23
            self.w24=New_w24 + self.w24
            self.theta4=New_theta4 + self.theta4
            Y3,Y4,Y5,e,self.gr5=self.initiateLayer()
            print(Y3,Y4,Y5,e,self.gr5)


n=NN(1,2,0.5,0.4,0.9,1,-1.2,1.1,0.8,-0.1,0.3)
n.gradientDesnt()



