# Silver(79th) solution for [tgs-salt-identification-challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)

I've tried many methods in this competition but mainly focus on seresnext-unet architecture.
In the src folder, there are bunch of notebooks using both keras and pytorch to implement the network.

## start with simple UNet

I started in unet_baseline.ipynb which defined a straight foward unet.
Then tried to fine tune it with deeper layers, se-blocks elu activation, kfolds.
Did not get much better results.

## Try a dedicated classifier
Since there are partial images with empty masks, it seems there maybe some rule to follow in identifying
the empty images.
I've tried transfer learning with all keras prebuild CNN classifiers such as DenseNet and InceptionResNetV2
in classifyer.ipynb.
They were some what working and after ensemble the results from classifiers and the unet segmentation model,
the results improved a little. Just a little.

## Try deeper UNets

In the next step, I tried deeper architectures in the backend of UNet with resblocks, inception blocks.
The results did not imrpove. The reason for this is covered until the end of this competition that the
organizer of this competions made some tricks in the train masks. Talk it later.

## Try to train a hybrid model

This is actually an old idea but works here.
It was to extract the backend output from previous Unet to make a classifier, then combine the output of
the classifier with the segementer to get the mix loss.
I tried to implement this idea in keras APIs with 2 loss but found it clumsy and the results nearly not improved.
Then I tried the pytorch with newer pretrained se_resnext50 from imagenet which results much better score in leaderboard.

## Kfolds with rounds

Then is the repeat training with kfolds and select the best model from the current round and train again.
The local CV iou increase a lot from the second round but the LB increase less.
I traind 5 folds for 4 rounds and get the best score before the competition ends.

## Just before it ends

The kaggle grandmaster Heng CherKeng explained why the label mask is whierd just hours before this competition
ends.
It is neccesary to try a Mosaics puzzle solving to really uncover this 'secret' and that's what I totally missed.
It says if combine one more channel to predict the edge of the salt into the previous model will bring the score in to 30th.
But I didn't have time to try it out.

![how the mask actually made](https://storage.googleapis.com/kaggle-forum-message-attachments/inbox/113660/8f78a30ee593a81693eb30ac0f129022/the%20real%20problem.png "how the mask actually made")



## Conclution 

There are always new skills to learn from each kaggle competitions and the masters in the platform like Heng CherKeng.
Keep fighting!
