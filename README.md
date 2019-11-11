# Nuclei-segmentation

Starter PyTorch implementation of U-Net (with skip-connections) image segmentation for Kaggle Data Science Bowl 2018 competition. Link to competition: https://www.kaggle.com/c/data-science-bowl-2018

# Insights

In the competition, most of the participants used various U-Net architectures. Here are some of the interesting takeaways from the competition:

### Modified train data

Winners of the competition released their solution and while they also used U-Net architecture, the main difference in their approach was that they manually(?) added increased loss to the area in-between nuclei that are nearby or are overlapping.

![labelled_loss](https://user-images.githubusercontent.com/16206648/68595706-44903e80-049a-11ea-9818-086be3f3f94a.png)

*Image 1. Example of how the **red marked areas** represent part of the image of increased loss by some factor*.

### Overfitting

Another issue that popped up was that it's really easy to overfit on it, even though a simple base U-Net model was used.

![](https://user-images.githubusercontent.com/16206648/68595692-3c380380-049a-11ea-9238-51ebb8efaddd.png)

*Image 2. Comparison of several models prediction (different training epochs) where **mask** (training label) was wrongly created*

### Generalization

While examining predicted samples, there was an indication that the models only learns how to differentiate white from black pixels. So I tested that...

![](https://user-images.githubusercontent.com/16206648/68595690-3c380380-049a-11ea-8a78-62f7caa3cff9.png)

### Conclusion

Yeah, there are some issues with current models and the training procedure... Probably adding regularization (dropout?) will somewhat alleviate the overfitting issues. As for generalization goes, it's difficult to say... Several things can be tried out: (1) More sophisticated creation of masks (each nucleus is labeled separately so when the masks are merged, overlapping nuclei lose shape), (2) some generalization will be obtained with reducing overfitting, (3) adding loss as the winners of the competition did or (4) data augmentation could help as well.
