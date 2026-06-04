Made a mlp in train1.py it is around 335k parameters.
In train.py increased the block size from 8 to 32 so parameters got increaced to 1.1million.
Use cuda(GPU) as it will give you results faster if your device not have gpu and ram is 8gb or less go with 16 block size only otherwise sometimes it may hang the system.
The output is still gibbrish as the model is understanding and predicting the next word but still not able to form the whole word and a sentence as it is a kind of limitation of the Multi Layered Perceptron.
So will be using the attention heads and make a transformer architecture and psuhing here updating this.
BYE😊
