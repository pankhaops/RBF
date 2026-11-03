import math

def gaussian(distance, sigma):
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))

x1 = int(input("Enter Input x1 (0 or 1): "))
x2 = int(input("Enter Input x2 (0 or 1): "))

centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
sigma = 1.41
weights = [-0.8, 0.9, 0.9, -0.8]

print("\n========== RBF Neural Network ==========")

print("\nStep 1 : Inputs")
print("x1 =", x1)
print("x2 =", x2)

hidden = []

print("\nStep 2 : Gaussian Activation")

for i in range(4):
    c1, c2 = centers[i]
    distance = math.sqrt((x1 - c1) ** 2 + (x2 - c2) ** 2)
    phi = gaussian(distance, sigma)
    hidden.append(phi)

    print(f"\nHidden Neuron {i+1}")
    print("Center =", centers[i])
    print("Distance =", round(distance, 4))
    print("Activation =", round(phi, 4))

print("\nStep 3 : Hidden Layer Outputs")

for i in range(4):
    print(f"H{i+1} = {hidden[i]:.4f}")

print("\nStep 4 : Weighted Sum")

weighted_sum = 0

for i in range(4):
    value = hidden[i] * weights[i]
    weighted_sum += value
    print(f"H{i+1} × W{i+1} = {hidden[i]:.4f} × {weights[i]} = {value:.4f}")

print("\nFinal Weighted Sum =", round(weighted_sum, 4))

print("\nStep 5 : Classification")

if weighted_sum >= 0:
    output = 1
else:
    output = 0

print("Predicted Output =", output)

actual = x1 ^ x2
print("Expected XOR Output =", actual)
