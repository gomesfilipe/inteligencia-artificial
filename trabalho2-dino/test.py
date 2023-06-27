def sum2Particles(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def sum3Particles(a, b, c):
    return [a[i] + b[i] + c[i] for i in range(len(a))]
    
def subParticles(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def multParticle(a, constant):
    return [constant * i for i in a]


x = [1,1,1,1]
v = [3,4,5,6]
y = [10,11,12,13]

print(multParticle(y, 3))
