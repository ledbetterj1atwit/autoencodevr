# Notes about the autoencoder.

At about 1-5 epochs it just, decides that not rendering green is a pretty good. I disagree.

Its still pretty bad but at 0.04 compression ratio I'm not sure I can complain...

Running on CPU sucks... but now I have CUDA!

Took a while to figure out how to get it to output

18 epochs: is it bottoming out? or just trying to find some new method?

Made model B: its model A minus the middle max pool and upsample layers, reduces compression to 0.16 compression...
But it does make it run pretty well, better on loss and traing is mostly unchanged. Ran this for 50 epochs while me and michael did homework.

Sunday now, running a model C(Model B but conv layer 2 has 12 finters instead of jsut 8), running it for 15 epochs.
So far at epoch 10 its faster on train but its leveling out now, time through the model is bigger though(35ms compared to 15ms on modelB).
Way better on train, less bouncy, but the test has already leveled off and its worse than B, hmm.
On the plus side, it has about as fast(maybe faster) time through the finished model, round 13ms.
