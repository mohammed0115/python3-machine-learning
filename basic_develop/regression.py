# Program to add two matrices using list comprehension
import numpy as np

X = np.array([[1,10,10,8],
    [1,0,8,10],
    [1,8 ,10,10],
[1,10 ,10,10]])
  

Y = np.array([[5000],
    [6000],
    [4000],
    [10000]])

hh=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
print(hh)

print(np.dot(np.array([[1,10,0,10]]),hh))
print(len(X[0]))
x_avarage=[]
realx=np.mean(X,axis=0)
for i in range(0,len(X[0])):
     x_avarage.append(np.mean(X,axis=0))
x_avarage=np.array(x_avarage)
print(x_avarage)
print("X**2=",X**2)
print("(x_avarage **2)=",(x_avarage **2))
print("(X **2) - (x_avarage **2)=",(X **2) - (x_avarage **2))
data=(X **2) - (x_avarage **2)
Sxx=np.sqrt(np.array(np.sum(data)))
print("Sxx=",Sxx)

y_avarage=[]
for i in range(0,len(Y)):
     y_avarage.append([np.mean(Y)])
y_avarage=np.array(y_avarage)
print(y_avarage)
print("Y**2=",Y**2)
print("(y_avarage **2)=",(y_avarage **2))
print("(y **2) - (y_avarage **2)=",(Y **2) - (y_avarage **2))
data=(Y **2) - (y_avarage **2)

Syy=np.sqrt(np.array(np.sum(data)))
print("Syy=",Syy)

data=np.sum(Y - y_avarage) * np.sum(X -x_avarage)
print("np.sum(Y - y_avarage) * np.sum(X -x_avarage)=",data)

Sxy=data
print("Sxy=",Sxy)

print("Sxx=",Sxx)
print("Syy",Syy)
correlation=Sxy/(Sxx * Syy)
print(np.sqrt(correlation))
#Sxx
#Syy
#Sxy
#(x*x~)^-1 *(x~*y)

