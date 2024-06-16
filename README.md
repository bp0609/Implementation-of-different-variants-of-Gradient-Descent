# Implementation-of-different-variants-of-Gradient-Descent
This repository contains implementations of various gradient descent algorithms, including stochastic gradient descent(SGD), full-batch gradient descent, and gradient descent with momentum. It also includes visualizations of the convergence process for each variant, facilitating comparison and better understanding of their performance and behavior.

## Gradient Descent
Gradient Descent is a fundamental optimization algorithm used to minimize the loss function in machine learning and statistical models. It iteratively adjusts the model parameters in the direction of the steepest decrease in the loss function, as determined by the gradient. The algorithm begins with an initial set of parameters and updates them by taking steps proportional to the negative of the gradient of the loss function at the current parameters. This process continues until the algorithm converges to a minimum value of the loss function, which could be a local or global minimum. Gradient Descent is widely used due to its simplicity and effectiveness in training a variety of machine learning models.

##  Generate the following two functions:

Dataset 1:
```python
num_samples = 40
np.random.seed(45) 
    
# Generate data
x1 = np.random.uniform(-20, 20, num_samples)
f_x = 100*x1 + 1
eps = np.random.randn(num_samples)
y = f_x + eps
```
<img src="plots/D1 contour_surface.png" width="600">
<br><br><br>

Dataset 2: 
```python
np.random.seed(45)
num_samples = 40
    
# Generate data
x1 = np.random.uniform(-1, 1, num_samples)
f_x = 3*x1 + 4
eps = np.random.randn(num_samples)
y = f_x + eps
```
<img src="plots/D2 contour_surface.png" width="600">
<br><br><br>

## Implementation of full-batch and stochastic gradient descent. 
Implement FBGD and SGD. Find the average number of steps it takes to converge to an $\epsilon$-neighborhood of the minimizer for both datasets. Visualize the convergence process for 15 epochs. Choose $\epsilon = 0.001$ for convergence criteria. Which dataset and optimizer takes a larger number of epochs to converge, and why? Show the contour plots for different epochs (or show an animation/GIF) for visualisation of optimisation process. Also, make a plot for Loss v/s epochs. <br><br>

## Full Batch Gradient Descent(FBGD)
Full-Batch Gradient Descent (FBGD) computes the gradient of the loss function using the entire dataset at each iteration. This method ensures a precise direction for updating the model parameters, providing a stable and consistent convergence path.
``` python
def full_batch_gradient_descent(X, y, initial_theta=-1, learning_rate=0.01, epsilon=0.001, max_epochs=100):
    # Initialize parameters randomly if initial_theta is not provided
    if initial_theta == -1:
        theta = np.random.randn(X.shape[1])
    else:
        theta = initial_theta

    # Add bias term to X if necessary
    if X.ndim == 1:
        X = np.hstack((np.ones((X.shape[0], 1)), X))

    count_epochs = 0
    loss_history = [loss_function(theta, X, y)]
    prev_loss = loss_history[0]
    theta_history = [theta.copy()]

    # Iterate over epochs
    for _ in range(max_epochs):
        # Compute the gradient
        gradient = np.dot(X.T, (np.dot(X, theta) - y)) / len(y)
        # Update parameters
        theta -= learning_rate * gradient
        # Calculate the loss
        loss = loss_function(theta, X, y)
        loss_history.append(loss.copy())
        theta_history.append(theta.copy())
        count_epochs += 1
        # Check for convergence
        if abs(loss - prev_loss) < epsilon:
            break
        prev_loss = loss.copy()

    return theta, loss_history, theta_history, count_epochs

```
<br>

## Stochastic Gradient Descent(SGD)
Stochastic Gradient Descent (SGD) updates the model parameters by computing the gradient based on a single training example at each iteration. This approach allows for faster updates and is well-suited for handling large datasets efficiently, though it introduces more variability in the convergence path.
```python
def stochastic_gradient_descent(X, y, initial_theta=-1, learning_rate=0.01, epsilon=0.001, max_epochs=100):
    # Initialize parameters randomly if initial_theta is not provided
    if initial_theta == -1:
        theta = np.random.randn(X.shape[1])
    else:
        theta = initial_theta

    # Add bias term to X if necessary
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))

    count_epochs = 0
    loss_history = [loss_function(theta, X, y)]
    prev_loss = loss_history[0]
    theta_history = [theta.copy()]

    # Iterate over epochs
    for _ in range(max_epochs):
        for i in range(len(y)):
            # Compute the gradient for the current sample
            gradient = np.dot(X[i].T, (np.dot(X[i], theta) - y[i]))
            # Update parameters
            theta -= learning_rate * gradient
        # Calculate the average loss for the epoch
        loss = np.mean((np.dot(X, theta) - y) ** 2) / 2
        loss_history.append(loss.copy())
        theta_history.append(theta.copy())
        count_epochs += 1
        # Check for convergence
        if abs(loss - prev_loss) < epsilon:
            break
        prev_loss = loss

    return theta, loss_history, theta_history, count_epochs

```
<br>
<br>

- Dataset 1  [Initalised ${\theta}_0$ =100  and ${\theta}_1=125$ , ${\alpha}=0.01$]

    -  Number of epochs taken by SGD to converge = 13

     - Number of epochs taken by FBGD to converge = 620

- Dataset 2 [Initalised ${\theta}_0$ =100  and ${\theta}_1=100$ , ${\alpha}=0.01$]

     - Number of epochs taken by SGD to converge =  59

     - Number of epochs taken by FBGD to converge =  1633

  ---
- Dataset 1  [Initalised ${\theta}_0$ =50  and ${\theta}_1=50$ , ${\alpha}=0.01$]

     - Number of epochs taken by SGD to converge = 12

     - Number of epochs taken by FBGD to converge = 546

- Dataset 2 [Initalised ${\theta}_0$ =50  and ${\theta}_1=50$ , ${\alpha}=0.01$]

     - Number of epochs taken by SGD to converge =  48

     - Number of epochs taken by FBGD to converge =  1396

  ---
- Dataset 1  [Initalised ${\theta}_0$ =25  and ${\theta}_1=0$ , ${\alpha}=0.01$]

     - Number of epochs taken by SGD to converge = 12

     - Number of epochs taken by FBGD to converge = 469

- Dataset 2 [Initalised ${\theta}_0$ =25  and ${\theta}_1=0$ , ${\alpha}=0.05$]

     - Number of epochs taken by SGD to converge =  5

     - Number of epochs taken by FBGD to converge =  452

  ---
    We know that FBGD and SGD are two optimization algorithms used to minimize the loss function.

    We can observe that SGD converges very fast compared to FBGD in both the datasets. This is because in FBGD, the entire dataset is used to compute the gradient of the loss function with respect to the model parameters in each iteration. This means that FBGD calculates the average gradient across the entire dataset. In contrast, SGD uses only a single randomly chosen data point to compute the gradient at each iteration. 

    SGD updates the model parameters more frequently using a single data point leading to more frequent but noisy updates. However, these noisy updates helps the algorithm escape local minima and saddle points more easily, leading to faster convergence.

    Near the global minima it is seen that the gradient vectors of individual loss function are along different direction and taking average results in very smaller updation of theta's.

<img src="plots/D1 lossVsepochs.png" width=400>
<br>

<img src="plots/D2 lossVsepochs.png" width=400>
    
We also see that dataset1 $(y=100x+1)$ converges fast than dataset2 $(y=3x+4)$ in this case. But the actual convergence speed depends on the initialisation and learning rate that we use for before applying gradient descent.

## Part 2 Momentum
------- 
   - Explore the article [here](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/#:~:text=Momentum%20is%20an%20extension%20to,spots%20of%20the%20search%20space.) on gradient descent with momentum. Implement gradient descent with momentum for the above two datasets. Visualize the convergence process for 15 steps. Compare the average number of steps taken with gradient descent (both variants -- full batch and stochastic) with momentum to that of vanilla gradient descent to converge to an $\epsilon$-neighborhood of the minimizer for both datasets. Choose $\epsilon = 0.001$. Write down your observations. Show the contour plots for different epochs for momentum implementation. Specifically, show all the vectors: gradient, current value of theta, momentum, etc.

   - Dataset 1

        Number of epochs taken by SGD_M to converge = 13

        Number of epochs taken by FBGD_M to converge = 562


   - Dataset 2

        Number of epochs taken by SGD_M to converge = 49  

        Number of epochs taken by FBGD_M to converge =  1485
    
We can observe that momentum gradient descent converges faster than normal gradient descent.
- Reason

    In normal gradient descent we use the gadient of objective function to navigate the search space. 
    But gradient descent may bounce around the search space that have large gradient or noisy gradient and can stuck in local minima. 
    Momentum gradient descent uses inertia to overcome such problems.Since momentum accumulates past gradients, it helps the optimization process to build up speed and escape from local minimas more efficiently.

    Momentum effectively acts as an adaptive step size in the optimization process. When the gradients consistently point in the same direction, momentum accelerates convergence by increasing the step size. When the gradients change direction, momentum helps to smooth out the updates, preventing overshooting.

Updation in Gradient Descent
```math
{\theta}_{i}={\theta}_{i-1} - {\alpha} \frac{\partial L}{\partial \theta} 
```
Updation in Momentum Gradient Descent 
$$newchange={\alpha} \frac{\partial L}{\partial \theta }+{\beta}*change $$
```math
{\theta}_{i}={\theta}_{i-1} - newchange
```
```math
=> {\theta}_{i}={\theta}_{i-1} - ({\alpha} \frac{\partial L}{\partial \theta }+{\beta}*change)
```
<img src="plots/D1 variation of theta.png" width = 800, hight = 800>

<img src="plots/D2 variation of theta.png" width = 800, hight = 800><br><br>


Dataset 1

|Method   | ${\theta}_0$  | ${\theta}_1$ |
| :------------ |:---------------:| -----:|
| FBGD      | 1.22547 | 100.00304 |
| SGD       | 1.01356        |   100.00015 |
| FBGDM | 1.12376        |    100.00288 |
| SGDM | 1.01527        |    100.00021 |
                
----
<br>
Dataset 2

| Method  | ${\theta}_0$  | ${\theta}_1$ |
| :------------ |:---------------:| -----:|
| FBGD      | 4.08830 | 3.72411 |
| SGD       | 4.0095        |   3.07433 |
| FBGDM | 4.08363        |    3.68577 |
| SGDM | 4.01651        |    3.12932 |
                
----

<img src="plots/D1 momentum of theta.png" width=800 hight = 800>

<br>
We can see that of momentum of theta0 is nearly zero and for theta 1 it is very high initially. 

<br>
<br>

<img src="plots/D2 momentum of theta.png" width=800 hight = 800>

We can see similar phenomenon in case of SGDM case also ,but the frequency of variations of the momentum is higher in SGDM compared to FBGDM 

In both the methods SGDM and FBGDM the momentum that drives ${\theta}_0$ is almost zero and therefore we does not see any inprovement in the convergence process for ${\theta}_0$

SGDM has higher momentum than FBGDM and therefore it converges faster than FBGDM

### Effect of ${\beta}$
- SGD with momentum

<img src="plots/D2 lossVsepoch SGDM.png" width=400 hight = 400>

Loss is in logarithmic scale
<br>


- FBGD with momentum

<img src="plots/D2 lossVsepoch FBGDM.png" width=400 hight = 400>
