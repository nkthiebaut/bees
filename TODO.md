
Important:
+ Implement model loading
+ Meta model:
    + Use predictions from several models (AlexNet, VGG, SVM)
    + Add images meta informations (image variance) 

Less important:
+ Try zero regularization and smaller fully connected layers
+ Network :  
    + Try drop-out at every level
    + Leaky rectifiers vs standard ReLU
    + Fix slowness when stride is not zero
+ Use re-trained networks
+ Normalize accross channels instead of per channel
+ Make flip optional in DataAugmentationBatchIterator
