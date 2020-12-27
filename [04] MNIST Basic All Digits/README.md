# [04] MNIST Basic

## Disclaimer
The objective of this learner is to use deep learning to recognize handwritten digits with an error rate below 1%. 

Basic tutorial on how to recognize number 3 and 7 is on [Ch04. MNIST Basic](https://colab.research.google.com/github/fastai/fastbook/blob/master/04_mnist_basics.ipynb)

Github assets by Melisa Surja

## Source code
My Jupyter Notebook file can be viewed here:

https://colab.research.google.com/drive/1yv0wLv6k0uvFhinBFC8DAExuEWpeqWL3?usp=sharing

## Questionnaire
1. How is a grayscale image represented on a computer? How about a color image?

> 0-255, 0 is white 255 is black. For color images, they have 0-255 value for r,g,b channels.

2. How are the files and folders in the MNIST_SAMPLE dataset structured? Why?
> `train` and `valid`. Because it's easy to classify and use them this way.

3. Explain how the "pixel similarity" approach to classifying digits works.

Find out the ideal image from all the images in same category. For example, stack all '3' images together and use mean() to find the average value for each pixel.

4. What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.
```python
li = [1,2,3,4]
li = list(filter(lambda x: x%2 == 1, li))
li = list(map(lambda x: x*2, li))
```

5. What is a "rank-3 tensor"?

Tensor with 3 dimensions/axes. Like a list of matrices.

6. What is the difference between tensor rank and shape? How do you get the rank from the shape?

Tensor shape consists of the length of each dimension. Rank is the total of dimensions/axes, which is the length of shape.

7. What are RMSE and L1 norm?

L1 = Mean Absolute Error
RMSE = Mean Squared Error

8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?

By using numpy or pytorch functions.

9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
```python
num = tensor(
[
	[1,2,3],
	[4,5,6],
	[7,8,9]
])
num *= 2
num[1:,1:]
```

10. What is broadcasting?

PyTorch's ability to perform mathematic functions on tensors with different ranks.

11. Are metrics generally calculated using the training set, or the validation set? Why?

Validation set, because training set has been used to train the model. We need new, unused data to know the error rate/accuracy.

12. What is SGD?

Stochastic Gradient Descent

13. Why does SGD use mini-batches?

Because calculating loss on all training data set is too time consuming, so it's done per batches. A gradient descent step is updated on this batch, instead of on epoch.

14. What are the seven steps in SGD for machine learning?

Initialize, Predict, Loss, Gradient, Step (Repeat to Predict), Stop

15. How do we initialize the weights in a model?

Random values are fine.

16. What is "loss"?

Testing the effectiveness of current parameters by calculating the difference between prediction and target. Automated Learning.

17. Why can't we always use a high learning rate?

Because if it's too high then the loss might jump around instead of steadily decreasing.

18. What is a "gradient"?

Derivative of a slope / loss function.

19. Do you need to know how to calculate gradients yourself?

No, just use PyTorch API

20. Why can't we use accuracy as a loss function?

Because accuracy is a true/false and not a function.

21. Draw the sigmoid function. What is special about its shape?

It only has value of 0-1

22. What is the difference between a loss function and a metric?

Loss function is used to recalculate the parameters of activation function (learning process), metric is used to evaluate the whole algorithm.

23. What is the function to calculate new weights using a learning rate?
```python
w -= gradient(w) * lr
```

24. What does the DataLoader class do?

Load all the data, apply transforms or augmentations, split training and validation sets, etc. Prepare the data.

25. Write pseudocode showing the basic steps taken in each epoch for SGD.
```python
def step(params):
	preds = f(x)
	loss = mse(preds, target)
	loss.backward()
	params.data -= lr * params.grad.data
	params.grad = None
	return preds
```

26. Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?
```python
list(zip([1,2,3,4],['a','b','c','d']))
```

27. What does view do in PyTorch?

PyTorch method that changes the shape of a tensor without changing its contents.

28. What are the "bias" parameters in a neural network? Why do we need them?

It's a function to create a line: `f(x): xw + b`

29. What does the @ operator do in Python?

Matrix multiplication.

30. What does the backward method do?

Calculate a function parameter's gradient.

31. Why do we have to zero the gradients?

Because loss.backward() adds to the current gradient value.

32. What information do we have to pass to Learner?

Dataset, Deep Learning Architecture, metrics (if any)

33. Show Python or pseudocode for the basic steps of a training loop.
```python
def train_epoch(model,lr,params):
	for xb, yb in dl:
	calc_grad(xb, yb, model)
	for p in params:
		p.data -= lr*p.grad.data
		p.grad.zero_()
```

34. What is "ReLU"? Draw a plot of it for values from -2 to +2.
```python
[0,0,0,1,2]
```

35. What is an "activation function"?

Function to create prediction.

36. What's the difference between F.relu and nn.ReLU?

nn.ReLU is a module, F.relu is a function.

37. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?

It's more complex
