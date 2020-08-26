
**Pending Items:**

1. Run the code get best hyper paras
		L1 + BN - Pavan
		L2+ BN - Ezhriko
		L1+ L2+ BN - Varsha
1. Add GBN to the code and create a separate model class - Pavan
2. Add the function which takes inputs of model and its accuracies & its losses as input and drew a graph - Ezhriko
3. Finally run all 5 models using best hyper paras from step 1 and use above function to draw the graph - Varsha
4. Misclassified images by GBN - We will see at the end

FYI - There is a method to save model written in shared notebook by me in case want to save the model in case of GBN model

**Hyper parameter search approach:**

Different Algorithms (l1, l2, l1&l2) with BN & l1&l2 with GBN
	Multiple ranges ([0-1],[0-0.01],[0-0.001], [0-0.0001])
		multiple values in each range
			multiple epochs
			Get best results in all epochs
		get best results in all values of specific range
	Get best results over all ranges

With above approach Coarse and Finer search  4 * 4 * 20 * 10 = 3200 epochs overall to get best parameters for 4 models (l1, l2, l1&l2) with BN & l1&l2 with GBN

4 - number of models
4 - number of ranges
20 - number of random values per range
10 - number of epochs

Total 3200 epochs to get best parameters to run even we keep number of epochs as 10.
