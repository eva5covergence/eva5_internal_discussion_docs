1. Choose best train and test pair and its hyper parameters 

==> Coarse Search

Each L1/L2 value - 

Choose best train and test acc in 25 epochs

	- Arrange the test accuracies in descending order
	- Eliminate the train and test accuracies pair if the train_acc more test_acc more than 1
	- Choose top 1 pair

If we follow above process for 20 values we get 20 top 1 Pairs

==> Finer Search

Each L1/L2 value - 

Choose best train and test acc in 25 epochs

	- Arrange the test accuracies in descending order
	- Eliminate the train and test accuracies pair if the train_acc more test_acc more than 1
	- Choose top 1 pair

If we follow above process for 20 values we get 20 top 1 Pairs

==> Final result from Finer Search 
	- This need to be done separately for L1+BN
	- This need to be done separately for L2+BN
	- This need to be done separately for L1+ L2 + BN

2. Create a model GBN


==> Final result from Finer Search 
	- This need to be done separately for L1+ L2 + GBN


Finally
3. Now we have best l1 and l2 paras for all 5 models

- Store the train, test acc and train and test losses for 5 models
- Drew the all curves of acc graph and loss graph over epochs of 25

Note if its GBN save the best model 

	- Check test accuracy —> if current_test_acc > best_acc :
							save model with respective epoch number etc…


