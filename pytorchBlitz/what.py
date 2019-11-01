import torch
x = torch.empty(5,3)
print(x)
print(torch.rand(5,3))
print(torch.zeros(5,3, dtype=torch.float))
print(torch.tensor([5.5,3]))
print(x.size())
y = torch.rand(5,3)
print(x+y)
print(torch.add(x,y))
result=torch.zeros(5,3)
torch.add(x,y,out=result)
print(result)
x.add_(y)
print(x)
print(x[:,-1])
#pytorch and numpy
#pytorch and numpy will share the memory location of the tensors if they're both on CPU
x_numpy =x.numpy()
x.add_(1)
print(x)
print(x_numpy)
# pytorch and cuda
if torch.cuda.is_available():
    device=torch.device("cuda")
    y = torch.zeros(device=device, size=(5,3))
    x = x.to(device)
    z = x+y
    print(z.to("cpu",torch.double))
