import numpy as np
m=0.5
b=20
x=np.array([2,3,5,6,8])
y=np.array([65,70,75,85,90])
def regression(x,m,b):
    return m*x+b
def loss(y,y_pred):
    return np.mean((y-y_pred)**2)
def update(x,y,y_pred,m,b,lr=0.01):
    dm=-2*np.mean(x*(y-y_pred))
    db=-2*np.mean(y-y_pred)
    new_m=m-lr*dm
    new_b=b-lr*db
    return new_m,new_b
loss_history=[]
for i in range(1000):
    #make prediction
    y_pred=regression(x,m,b)
    #calculate loss
    loss_value=loss(y,y_pred)
    loss_history.append(loss_value)
    
    if i%100==0:
        print(f"Iteration {i}: Loss={loss_value}, m={m}, b={b}")
    #update parameters
    m,b=update(x,y,y_pred,m,b)
print("\n---Training complete---")
print(f"final loss:{loss_history[-1]:.2f}")
