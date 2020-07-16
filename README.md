# Malaria-with-Deep-Learning
Classification of parasitic and normal cell and prediction using Deep Learning models

# About the Dataset
I have used publicly available datasets from [Kaggle](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

The Dataset consists of cell images of types:-
1. Parasitized
2. Uninfected

And a total of 27,558 images.
# Cell images

![parasitized](https://www.kaggleusercontent.com/kf/38019995/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ci9NK3FUepZwRjYXra-tPQ.9vgHR-WpY2BbjkbACV3npShfZsbr4gB6yW76O1iMisZRMkxJ_B1ZeVzlVOT5ebnITCiU2hq2fVNKJPEGMkD5u0qhVlohHHwnP-4Z_yxcpoONAPhq7mAcQcSb2VOO4Fci2LBJPWWvAX61CF0K5-cZfD7ATcNX0qCZAOlQyUtt7AyGwqR5VmnDjTE5GuXruHTFIy3t-tsqnaLCYR2rMYEdKf3Pvm54g5laOjSuiytVsZDRiIxBPu1x_Y_1_22DIkB90MJvBQsLbCRh17xHrIaKlZQIA6QNGAf8w-ERdeAF0wBKCRVA8OzDntSElVGYRirVAOUAl2uxoq6aYg0pqoTGWXt3AilwsL_-agkpIItk-HDlReFe-XEzUIr8Hnb2jhnONV8jNzFvzMQIyFaAUdYIr6Myf3AyAETbbQDtP9xOEzcADObC_GIywjouEKaam9EV_xBpYnYPWkXA-77r5jA4D72RFYf_uXsk_bXY9ngxQSLO-Cjf4s3_jYzkM4Mcm1jt6SJsY33kVaNqY2Cc-_T_xzyPr9nXOZ4WMQKKHW0ufxWwu-wstd2DECpFA3UUqRxIVy9QOqzXUaQ3iX5XmyahAlyTjOCx19mvgJ_wlHiT5_njwdGhClocCBm8xC4xF08ev0p0_ESFIAkUpo9x8tseVjZ1t_ZOmh6wo5Gsoi9_E1V1IHP9134-Uw2dr6gzEBu7.c3F46bAKP-_yj9H2xuDmYQ/__results___files/__results___9_1.png)

# Project Layout
I have used [MobileNetV2](https://keras.io/api/applications/mobilenet/), [VGG19](https://keras.io/api/applications/vgg/) and [InceptionNetV3](https://keras.io/api/applications/inceptionv3/) transfer learning model for simple image classification and predictions

**Other Libraries**
1. [Tensorflow](https://tensorflow.org/)
2. [Matplotlib](https://matplotlib.org/) (for Visualizations)
3. [Sklearn](https://scikit-learn.org/stable/) for confusion matrix and classification reports

[ImageDataGenerator](https://keras.io/api/preprocessing/image/) of keras was used for classifying and augmenting the images for model based predictions

# Metrics Plot

![accuracy](https://www.kaggleusercontent.com/kf/38019995/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ci9NK3FUepZwRjYXra-tPQ.9vgHR-WpY2BbjkbACV3npShfZsbr4gB6yW76O1iMisZRMkxJ_B1ZeVzlVOT5ebnITCiU2hq2fVNKJPEGMkD5u0qhVlohHHwnP-4Z_yxcpoONAPhq7mAcQcSb2VOO4Fci2LBJPWWvAX61CF0K5-cZfD7ATcNX0qCZAOlQyUtt7AyGwqR5VmnDjTE5GuXruHTFIy3t-tsqnaLCYR2rMYEdKf3Pvm54g5laOjSuiytVsZDRiIxBPu1x_Y_1_22DIkB90MJvBQsLbCRh17xHrIaKlZQIA6QNGAf8w-ERdeAF0wBKCRVA8OzDntSElVGYRirVAOUAl2uxoq6aYg0pqoTGWXt3AilwsL_-agkpIItk-HDlReFe-XEzUIr8Hnb2jhnONV8jNzFvzMQIyFaAUdYIr6Myf3AyAETbbQDtP9xOEzcADObC_GIywjouEKaam9EV_xBpYnYPWkXA-77r5jA4D72RFYf_uXsk_bXY9ngxQSLO-Cjf4s3_jYzkM4Mcm1jt6SJsY33kVaNqY2Cc-_T_xzyPr9nXOZ4WMQKKHW0ufxWwu-wstd2DECpFA3UUqRxIVy9QOqzXUaQ3iX5XmyahAlyTjOCx19mvgJ_wlHiT5_njwdGhClocCBm8xC4xF08ev0p0_ESFIAkUpo9x8tseVjZ1t_ZOmh6wo5Gsoi9_E1V1IHP9134-Uw2dr6gzEBu7.c3F46bAKP-_yj9H2xuDmYQ/__results___files/__results___16_0.png)

![loss](https://www.kaggleusercontent.com/kf/38019995/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..ci9NK3FUepZwRjYXra-tPQ.9vgHR-WpY2BbjkbACV3npShfZsbr4gB6yW76O1iMisZRMkxJ_B1ZeVzlVOT5ebnITCiU2hq2fVNKJPEGMkD5u0qhVlohHHwnP-4Z_yxcpoONAPhq7mAcQcSb2VOO4Fci2LBJPWWvAX61CF0K5-cZfD7ATcNX0qCZAOlQyUtt7AyGwqR5VmnDjTE5GuXruHTFIy3t-tsqnaLCYR2rMYEdKf3Pvm54g5laOjSuiytVsZDRiIxBPu1x_Y_1_22DIkB90MJvBQsLbCRh17xHrIaKlZQIA6QNGAf8w-ERdeAF0wBKCRVA8OzDntSElVGYRirVAOUAl2uxoq6aYg0pqoTGWXt3AilwsL_-agkpIItk-HDlReFe-XEzUIr8Hnb2jhnONV8jNzFvzMQIyFaAUdYIr6Myf3AyAETbbQDtP9xOEzcADObC_GIywjouEKaam9EV_xBpYnYPWkXA-77r5jA4D72RFYf_uXsk_bXY9ngxQSLO-Cjf4s3_jYzkM4Mcm1jt6SJsY33kVaNqY2Cc-_T_xzyPr9nXOZ4WMQKKHW0ufxWwu-wstd2DECpFA3UUqRxIVy9QOqzXUaQ3iX5XmyahAlyTjOCx19mvgJ_wlHiT5_njwdGhClocCBm8xC4xF08ev0p0_ESFIAkUpo9x8tseVjZ1t_ZOmh6wo5Gsoi9_E1V1IHP9134-Uw2dr6gzEBu7.c3F46bAKP-_yj9H2xuDmYQ/__results___files/__results___16_1.png)

***Model had 93.7% accuracy on validation dataset!!***

**If you like this do upvote my [Notebook](https://www.kaggle.com/digvijayyadav/malaria-and-simple-deep-learning) as it will motivate me to make more interesting notebooks!!**

***Feel free to comment on my implementation, suggestions are always welcome! :)***

**Happy Learning!!**
# Acknowledgements
This Dataset is taken from the official NIH Website: https://ceb.nlm.nih.gov/repositories/malaria-datasets/
And uploaded here, so anybody trying to start working with this dataset can get started immediately, as to download the
dataset from NIH website is quite slow.
Photo by Егор Камелев on Unsplash
https://unsplash.com/@ekamelev

Inspiration
Save humans by detecting and deploying Image Cells that contain Malaria or not!
