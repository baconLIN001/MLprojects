__author__ = 'bacon'
import matplotlib.pyplot as plt
import numpy as np

def dual_perceptron(dataSet,N,a,b,n):
    flag = True
    while flag:
        for i in range(N):
            tmp=0
            for j in range(N):
                tmp += a[j] * dataSet[j][1] * np.dot(dataSet[j][0],dataSet[i][0])
            tmp += b
            tmp *= dataSet[i][1]
            print(tmp)
            if tmp <= 0:
                a[i] += n
                print("a:  %d",a)
                b += n * dataSet[i][1]
                print("a:  %d",b)
                break
            if i == N-1:
                flag = False
    return a,b

if __name__ == "__main__":
    N=7
    a=np.zeros(N,np.float32)
    b=0
    n=1
    x=[3,4,1,2,4,2,2]
    y=[3,3,1,0,4,2,3]
    # dataSet=np.array([x,y])
    data=[[[3,3],1],[[4,3],1],[[1,1],-1],[[2,0],-1],[[4,4],1],[[2,2],-1],[[2,3],-1]]
    # print(dataSet)
    print(data)
    plt.figure()
    plt.scatter(x,y)
    # print(data.pop(2).pop(1))
    # print(data.pop(1).pop(0))
    # for i in range(N):
    #     print a[i]
    #     print data[i]
    #     plt.plot(data[i][0][0],data[i][0][1])

    # m=[3,3]
    # n=[4,3]
    # print(np.dot(m,n))
    # dataSet=np.array([3,3],[4,3])
    a,b=dual_perceptron(data,N,a,b,n)
    print(a,b)
    w=np.zeros(2,np.float32)
    for j in range(N):
       # w = w + np.dot(a[j]*data[j][1],data[j][0][0])
       w[0]+=a[j]*data[j][1]*data[j][0][0]
       w[1]+=a[j]*data[j][1]*data[j][0][1]
       print(data[j][1],data[j][0][0],data[j][0][1])
    print(w)
    x1=np.arange(0,5)
    plt.plot(x1,(-b-w[0]*x1)/w[1])
    plt.show()
    # print(a)