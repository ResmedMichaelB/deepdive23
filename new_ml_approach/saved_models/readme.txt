------------------------------
position_model1
Date: 03/03/23
5 layer cnn model with number of filters doubling every 2nd layer
4 filters in 1st layer. Kernel size =3
Dense layer 16
Dropout = 0.3
Trained on s9 dataset
Trained for ~60 epochs
Batch size 64
learning rate = 0.0001
Max pooling

Window size 60, step size 30
Trained on flow only.
Some filtering involved in the signal. I think a second order butterworth, cut-off at 3hz

70% accuracy on the test set and validation set. 
F1 validation set = 70% at a threshold of 0.5

Notes: Very slow to converge but also quite stable in its increase. And didn't seem to overfit much 
I would be interested to see how far it could be pushed before serious overfitting

I'd also be interested to see how unfiltered data fairs
--------------------------------------------------------


