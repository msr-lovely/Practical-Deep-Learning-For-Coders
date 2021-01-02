# 6. Other Computer Vision Problems

## Questionnaire Answers
1. How could multi-label classification improve the usability of the bear classifier?
> It can identify "no bear at all" instead of always identifying either black, grizzly, or teddy bear.

2. How do we encode the dependent variable in a multi-label classification problem?
> Hot-encoded targets, the correct labels have value of `1` and the wrong labels have value of `0`

3. How do you access the rows and columns of a DataFrame as if it was a matrix?
```
df.iloc[x,y]
```

4. How do you get a column by name from a DataFrame?
```
df[column_name]
```

5. What is the difference between a `Dataset` and `DataLoader`?
> **Dataset**: A collection that returns a tuple of your independent and dependent variable for a single item.
> **DataLoader**: An iterator that provides a stream of mini-batches, where each mini-batch is a tuple of a batch of independent variables and a batch of dependent variables.

6. What does a `Datasets` object normally contain?
> Training + Validation Dataset.

7. What does a `DataLoaders` object normally contain?
> Training + Validation DataLoader.

8. What does `lambda` do in Python?
> Differences

9. What are the methods to customize how the independent and dependent variables are created with the data block API?
> `get_x` and `get_y`

10. Why is softmax not an appropriate output activation function when using a one hot encoded target?
> Because softmax sum is one, and it tends to make larger one activation over another because of exponential function.

11. Why is `nll_loss` not an appropriate loss function when using a one-hot-encoded target?
> It returns the value of just one activation.

12. What is the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?
> `nn.BCELoss` calculated cross-entropy on a one-hot-encoded target, but without the initial sigmoid.

13. Why can't we use regular accuracy in a multi-label problem?
> Because it only takes highest activation value, which means it only gives one label and can't give multiple labels.

14. When is it okay to tune a hyperparameter on the validation set?
> Because it didn't overfit, the curve of the threshold variables is very smooth, not jagged or volatile like a model that suffers from overfitting.

15. How is `y_range` implemented in fastai? (See if you can implement it yourself and test it without peeking!)
```python
def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo
```

16. What is a regression problem? What loss function should you use for such a problem?
> Learning from a dataset where the independent variable is an image, and the dependent variable is one or more floats.
> Loss function for regression = `nn.MSELoss`

17. What do you need to do to make sure the fastai library applies the same data augmentation to your input images and your target point coordinates?
> Use PointBlock in DataBlock API
