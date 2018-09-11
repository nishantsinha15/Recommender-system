import matplotlib.pyplot as plt

a = [10,20,30,40,50]
error = [0.7730616680276463, 0.7868596323607691, 0.79677381572535, 0.8022660208559269, 0.8054858031425983]
plt.plot(a, error )
plt.xlabel("Number of neighbours")
plt.ylabel("MAE")
plt.savefig('User user.png')