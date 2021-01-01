# 7. Training a State-of-the-Art Model 

## Questionnaire

1. What is the difference between ImageNet and Imagenette? When is it better to experiment on one versus the other?
> Imagenette is a smaller set of images from ImageNet. It is created for average people without an access to high-end hardware to prototype on. Imagenette was selected by 10 classes from the full ImageNet that looked very different from one another.

2. What is normalization?
3. Why didn't we have to care about normalization when using a pretrained model?
4. What is progressive resizing?
5. Implement progressive resizing in your own project. Did it help?
6. What is test time augmentation? How do you use it in fastai?
7. Is using TTA at inference slower or faster than regular inference? Why?
8. What is Mixup? How do you use it in fastai?
9. Why does Mixup prevent the model from being too confident?
10. Why does training with Mixup for five epochs end up worse than training without Mixup?
11. What is the idea behind label smoothing?
12. What problems in your data can label smoothing help with?
13. When using label smoothing with five categories, what is the target associated with the index 1?
14. What is the first step to take when you want to prototype quick experiments on a new dataset?

