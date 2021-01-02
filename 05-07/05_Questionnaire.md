# 5. Image Classification

## Questionnaire Answers

1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
> Because if we perform the augmented transformation on reduced size, then the images would have empty spaces when they're rotated/warped.

2. If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.
> I have used REGex extensively in the past :3

3. What are the two ways in which data is most commonly provided, for most deep learning datasets?
> 1. Individual files organized into folders. The folder's name is the category/label.
> 1. A table of data that provides connection between data.

4. Look up the documentation for L and try using a few of the new methods that it adds.
```python
p.filter(ge(15))
```
> Enables functional methods to filter the list.

5. Look up the documentation for the Python pathlib module and try using a few methods of the Path class.
```python
Path().ls() 
Path().exists()
```

6. Give two examples of ways that image transformations can degrade the quality of the data.
> - If the transformations are too extreme and the image is hard to recognize.
> - Cropping the image in multi-label classifications can crop out some objects that are located near the corners.

7. What method does fastai provide to view the data in a DataLoaders?
```python
dls.show_batch()
```

8. What method does fastai provide to help you debug a DataBlock?
```python
db.summary(path)
```

9. Should you hold off on training a model until you have thoroughly cleaned your data?
> Yes. To debug DataBlock, use `db.summary()`

10. What are the two pieces that are combined into cross-entropy loss in PyTorch?
> Predictions and Targets

11. What are the two properties of activations that softmax ensures? Why is this important?
> Softmax ensures that all activations are between 0-1 and that they all sum to 1.

12. When might you want your activations to not have these two properties?

13. Calculate the exp and softmax columns of <bear_softmax> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).
<img src='https://raw.githubusercontent.com/fastai/fastbook/3916b71bdf2f9e587ac82f3c2ef4aabd05b8f51c/images/att_00062.png' />

14. Why can't we use torch.where to create a loss function for datasets where our label can have more than two categories?
> Because torch.where can only map between two values, so it only works on two categories.

15. What is the value of log(-2)? Why?
> np.log(-2) returns an error (invalid value) because logarithm can't produce a negative value. For example, np.log(100) is 2, because 10**2 is 100.

16. What are two good rules of thumb for picking a learning rate from the learning rate finder?
> - One order of magnitude less than where the minimum loss was achieved (i.e., the minimum divided by 10)
> - The last point where the loss was clearly decreasing 

17. What two steps does the fine_tune method do?
> 1. Train the new random layer for one epoch
> 1. Train the model (all layers) for all the epoch

18. In Jupyter Notebook, how do you get the source code for a method or function?
> By using `??` before the method/function, for example:
```python
??show_image
```

19. What are discriminative learning rates?
> Use a lower learning rate for early layers of pretrained neural network, and a higher learning rate for the later layers (the new randomly added ones)

20. How is a Python slice object interpreted when passed as a learning rate to fastai?
> The first value passed will be the learning rate in the earliest layer of the neural network, and the second value will be the learning rate in the final layer.

21. Why is early stopping a poor choice when using 1cycle training?
> Because if you stop in the middle, the learning rate hasn't reached its minimum, which is where you get the best results.

22. What is the difference between resnet50 and resnet101?
> Resnet 101 has more layers and therefore more parameters.

23. What does to_fp16 do?
> Uses tensor cores that can speed up neural network training. Almost all NVIDIA GPUs support it. My speed didn't increase though.


