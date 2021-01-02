# 7. Training a State-of-the-Art Model 

## Questionnaire

1. What is the difference between ImageNet and Imagenette? When is it better to experiment on one versus the other?
> Imagenette contains 10 classes from the full ImageNet that looked very different from one another. It is created for average people without an access to high-end hardware to prototype on.

2. What is normalization?
> The process to equalize the scale that your input use with the scale that the pretrained model's input use. For example: the images that you use might have 0-255 value to represent their colors, but if you're using a model that was pretrained on ImageNet, you need to convert your images' 0-255 value to ImageNet's scale (mean of 0 and standard deviation of 1)
 
3. Why didn't we have to care about normalization when using a pretrained model?
> Because there's nothing to normalize to.

4. What is progressive resizing?
> In the beginning, use smaller images to train images for a few epochs. Then, use normal size to fine tune. It is better for model generalization and it's faster to train.

5. Implement progressive resizing in your own project. Did it help?
> Yes.

6. What is test time augmentation? How do you use it in fastai?
> Using data augmentation methods on validation set.
```python
preds,targs = learn.tta()
accuracy(preds,targs).item()
```

7. Is using TTA at inference slower or faster than regular inference? Why?
> Slower, because it needs more time for validation.

8. What is Mixup? How do you use it in fastai?
> Mixing the inputs and the labels. In case of images, stack two images together according to their weights.

9. Why does Mixup prevent the model from being too confident?
> Because it's a mix between two inputs instead of just one. It also automatically implements label smoothing.

10. Why does training with Mixup for five epochs end up worse than training without Mixup?
> Because Mixup is harder to train and needs far more epochs to get better accuracy.

11. What is the idea behind label smoothing?
> Smoothing the label's value from 0 and 1 to slightly more than 0 and slightly less than 1. This helps prevent the model from becoming overconfident.

12. What problems in your data can label smoothing help with?
> Helps with overfitting.

13. When using label smoothing with five categories, what is the target associated with the index 1?
> Target with index 1 is the correct category. Other categories with index 0 are the wrong ones. With label smoothing, the values wouldn't be 0 or 1, however.

14. What is the first step to take when you want to prototype quick experiments on a new dataset?
> If the new dataset is huge, don't use the whole dataset to prototype on. Train it on a small subset of representative of the dataset instead to iterate faster.
