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

------------------------------
position_model6
Date: 20/03/23
4 layer cnn model with number of filters doubling every 2nd layer
5 filters in 1st layer. Kernel size =10
Added strides for this. Trains much faster due to less complex model. SL = kernel size
Dense layer 16
Dropout = 0.3
Trained on s9 dataset and fillius dataset. The data wasn't shuffled. (Shuffling seemed to kill perfromance)
Test subjects were all s9 (to compare with other data)
Trained for 100 epochs
Batch size 64
learning rate = 0.0001
Max pooling

validation accuracy and training accuracy leveled off at 70
Due to unshuffled?

------------------------------
position_model7
Date: 22/03/23
Used Flow AND Pressure
4 layer cnn model with number of filters doubling every 2nd layer
5 filters in 1st layer. Kernel size =10
SL = kernel size
Dense layer 16
Dropout = 0.4
Trained on s9 dataset and fillius dataset. The data wasn't shuffled. (Shuffling seemed to kill perfromance)
Test subjects were all s9 (to compare with other data)
Trained for 100 epochs
Batch size 64
learning rate = 0.0001
Max pooling

Model overfit to training pretty quick. Also training performance was poor
Too large a model and pressure doesn't really do anything?

------------------------------
position_model8
Date: 24/03/23
Used Flow AND Pressure
Big change was using a window of 30 seconds with step size of 15.
4 layer cnn model with number of filters doubling every 2nd layer
5 filters in 1st layer. Kernel size =7
SL = kernel size
Dense layer 16
Dropout = 0.4
Trained on s9 dataset and fillius dataset. The data wasn't shuffled. (Shuffling seemed to kill perfromance)
Test subjects were all s9 (to compare with other data)
Trained for 100 epochs
Batch size 64
learning rate = 0.0001
Max pooling

Training performance plateaued at 65-67% whereas validation performance kept increasing.
Probably due to the fact that training dataset was a mix. Validation accuracy didn't seem to leveling 
So will give it another round

------------------------------
position_model9
Date: 27/03/23
Used Flow AND Pressure
Big change was using a window of 30 seconds with step size of 15.
4 layer cnn model with number of filters doubling every 2nd layer
5 filters in 1st layer. Kernel size =7
SL = kernel size
Dense layer 16
Dropout = 0.4
Trained on s9 dataset and fillius dataset. The data wasn't shuffled. (Shuffling seemed to kill perfromance)
Test subjects were all s9 (to compare with other data)
Trained for 200 epochs
Batch size 64
learning rate = 0.0001
Max pooling

Training performance plateaued at 68% whereas validation performance kept increasing. Validation accuracy F1 =80%! test =73%
Probably due to the fact that training dataset was a mix. Validation accuracy didn't seem to leveling 
So will give it another round

