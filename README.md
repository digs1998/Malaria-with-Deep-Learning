# Malaria-with-Deep-Learning
Classification of parasitic and normal cell and prediction using Deep Learning models

# About the Dataset
I have used publicly available datasets from [Kaggle](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

The Dataset consists of cell images of types:-
1. Parasitized
2. Uninfected

And a total of 27,558 images.
# Cell images

![parasitized](https://www.kaggleusercontent.com/kf/39382694/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..57hlAfg-hui2MOm9kHhCBg.Npk6XJvwl99CQkHq7fvY91UJX4fmjp1kzo2nw0BYEc4SkKF6qU9lrbP7FzGWsahRJ3ptXx3u3-b0aUqLz58JSIy0oqJfaaFO0vPCSQ4zDaz039YSK1C0MUoM3txGiUg1IWlryUR80uA3pRUxiLgHxyoiiCgWqQQichAKgMglULvImox53GgRqzwIyazMiauBX4ArwJY3-jNWVfTzGokx57Dd54qKI_WCRbUA_BY3uvf1S3ZsjaQ62hNE3BXJputDWjUXB9SGVZmVwF_5TBUyS7ij8xbpEGKlrYG-flrihiT9JPOo7Beq0rkX7AAFdNUH8viAB1gLT37gsabZPWTLdmUEgrnHXjxI9_1StFu-TlVLvpjUSB78nD15qHkrW52GGyR0sXOpxa-RQebBwOKqmnmPVuSi6X9BkCJaDPk9YCD4EQybnLMAM-vjkFzNTqHKDXyvmWcdbC71f6wLMF5_YjGJzBkfPVy4acv1iOJlUbly-qjr9Mj6x7a3R6MkU5ERImw3duK3M2dqRvunUsxft7KlN8lwBW883z9RX_9tMEXoWx8W0Tkti7yLYfSsOQ6K_7AHmE9mDNKws5hc0iAsY83keJ2dIRXxU3_KEugovydOYCyoiUXimvoL5PD0tvi_fmA2f6dBNpUU-wPb9FhyGd8sw6pp_ayP0pzk5IpDLPU2SQPBye8k34cC5hS_tYv_.OtOCnSaHo_WxwfR-miT3aQ/__results___files/__results___14_1.png)

# Project Layout
I have used [MobileNetV2](https://keras.io/api/applications/mobilenet/), [VGG19](https://keras.io/api/applications/vgg/) and [InceptionNetV3](https://keras.io/api/applications/inceptionv3/) transfer learning model for simple image classification and predictions

**Other Libraries**
1. [Tensorflow](https://tensorflow.org/)
2. [Matplotlib](https://matplotlib.org/) (for Visualizations)
3. [Sklearn](https://scikit-learn.org/stable/) for confusion matrix and classification reports

[ImageDataGenerator](https://keras.io/api/preprocessing/image/) of keras was used for classifying and augmenting the images for model based predictions

# Metrics Plot

![accuracy](https://www.kaggleusercontent.com/kf/38874612/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Uc-3f3emAAD4R1zPyopoOA.KVYlR7K-L5R1PQbD98aei7rFzlaoUOVOcalJgifLVHzL76iktj_YTFyuCls0lhoHt_q5JzWYyTHwgINMXlifG83_vvAFbrS9Q0ZWe1UbXXOEDzjPdVDmBtyxTyQLY7EtDva2P1muxKnadXIxUCbqhGuR7p2mvuD9J0ouPcShUEu79jwlC6hoZLIE-cAG9_MyReI6gsIffIxgW6OgwISRRL4mfvSdwdsqXjxTiEqjvOuzLlReAWMTrXfnzAZrC9VNtp3r29ZGcAoKF51iQHFaDTi5ejQ3KLFGk9Wd-hEdbN4GPv52JbtjQ53oxbkCpAVN1hwCVnAQI3NUaqDGDpct9uJcbyZEv-eyWEu0K-G2tV3_3oMzo6npycUxbzckytBoqFHMrlKn40mMKp6vw-Dag5kna17VkE9kalWQ8UP5WOxp1fga531lq_HXFFrmRgmRGO-Ts5pImAyFXUfbX_McVTdAfH_77jc-u6ay_IJvT0RVf8HfmEz0NBrN1AS1_m7jn-c6whW_Mnmr95e3yKDiAnaylzzDVIXYuOxrpN4YTdS0j8YwxdqRwJwm6-OEWaGktm1KASQKvQOXQXQ0lcXirhStC4j71Igw7vNtDTqL5dXQb42KEktzuW1J0DhWWDsKecCNVIBYouGEaQLpBv6j5upUZq51i08X7AcDZhJHjGM954nt1AXyx3WYZFpXGjfL.-yYO21VqqLIB5DNt9cfafQ/__results___files/__results___61_0.png)

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
