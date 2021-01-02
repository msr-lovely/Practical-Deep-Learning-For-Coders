# 5. Image Classification

## Questionnaire Answers

1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?
> 

2. If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.
> I have used REGex extensively in the past :3

3. What are the two ways in which data is most commonly provided, for most deep learning datasets?
> 1. Individual files

4. Look up the documentation for L and try using a few of the new methods that it adds.
>

5. Look up the documentation for the Python pathlib module and try using a few methods of the Path class.
>

6. Give two examples of ways that image transformations can degrade the quality of the data.
>

7. What method does fastai provide to view the data in a DataLoaders?
```
dls.show_batch()
```

8. What method does fastai provide to help you debug a DataBlock?
```
db.summary(path)
```

9. Should you hold off on training a model until you have thoroughly cleaned your data?
>

10. What are the two pieces that are combined into cross-entropy loss in PyTorch?
>

11. What are the two properties of activations that softmax ensures? Why is this important?
>

12. When might you want your activations to not have these two properties?
>

13. Calculate the exp and softmax columns of <bear_softmax> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).
<img src='https://raw.githubusercontent.com/fastai/fastbook/3916b71bdf2f9e587ac82f3c2ef4aabd05b8f51c/images/att_00062.png' />
> 

14. Why can't we use torch.where to create a loss function for datasets where our label can have more than two categories?
> Because torch.where can only map between two values, so it only works on two categories.

15. What is the value of log(-2)? Why?
> np.log(-2) returns an error (invalid value) because logarithm can't produce a negative value. For example, np.log(100) is 2, because 10**2 is 100.

16. What are two good rules of thumb for picking a learning rate from the learning rate finder?
>

17. What two steps does the fine_tune method do?
> 1. Train the new random layer for one epoch
> 1. Train the model (all layers) for all the epoch

18. In Jupyter Notebook, how do you get the source code for a method or function?
> By using `??` before the method/function, for example:
```
??show_image
```

19. What are discriminative learning rates?
> Learning rates start off high in the beginning and becomes lower at the end.

20. How is a Python slice object interpreted when passed as a learning rate to fastai?
>

21. Why is early stopping a poor choice when using 1cycle training?
>

22. What is the difference between resnet50 and resnet101?
> Resnet 101 has more layers and therefore more parameters.

23. What does to_fp16 do?


