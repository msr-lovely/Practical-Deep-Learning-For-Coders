# 6. Other Computer Vision Problems

## Questionnaire Answers
1. How could multi-label classification improve the usability of the bear classifier?
> It can identify "no bear at all" instead of always identifying either black, grizzly, or teddy bear.

1. How do we encode the dependent variable in a multi-label classification problem?
> Hot-encoded targets, the correct labels have value of `1` and the wrong labels have value of `0`

1. How do you access the rows and columns of a DataFrame as if it was a matrix?
```
df.iloc[x,y]
```

1. How do you get a column by name from a DataFrame?
```
df[column_name]
```

1. What is the difference between a `Dataset` and `DataLoader`?
1. What does a `Datasets` object normally contain?
1. What does a `DataLoaders` object normally contain?
1. What does `lambda` do in Python?
1. What are the methods to customize how the independent and dependent variables are created with the data block API?
1. Why is softmax not an appropriate output activation function when using a one hot encoded target?
1. Why is `nll_loss` not an appropriate loss function when using a one-hot-encoded target?
1. What is the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?
1. Why can't we use regular accuracy in a multi-label problem?
1. When is it okay to tune a hyperparameter on the validation set?
1. How is `y_range` implemented in fastai? (See if you can implement it yourself and test it without peeking!)
1. What is a regression problem? What loss function should you use for such a problem?
1. What do you need to do to make sure the fastai library applies the same data augmentation to your input images and your target point coordinates?