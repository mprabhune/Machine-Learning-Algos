# Machine-Learning-Algos
below ml models are implemented using Matlab:
linear regression
logistic regression
neural network
decision trees
principal component analysis

Commands to run:

neural network
neural_network("pendigits_training.txt" ,"pendigits_test.txt", 2, 0, 50)
neural_network("pendigits_training.txt" ,"pendigits_test.txt", 3, 20, 20)

decision tree
dtree("pendigits_training.txt", "pendigits_test.txt", "optimized", 50);
dtree("pendigits_training.txt", "pendigits_test.txt", "randomized", 50);
dtree("pendigits_training.txt", "pendigits_test.txt", "forest3", 50);
dtree("pendigits_training.txt", "pendigits_test.txt", "forest15", 50);

pca_power
pca_power("pendigits_training.txt", "pendigits_test.txt", 1, 10);
pca_power("satellite_training.txt", "satellite.txt", 2, 20);
pca_power("yeast_training.txt", "yeast_test.txt", 3, 30);

Logistic regression
ll_regression("pendigits_training.txt" , 1, "pendigits_test.txt")
ll_regression("pendigits_training.txt" , 2, "pendigits_test.txt")

linear regression
linear_regression("sample_data1.txt", 1,0)
linear_regression("sample_data1.txt", 2,0)
linear_regression("sample_data1.txt", 2,0.001)
linear_regression("sample_data1.txt", 2,1)

inputs files are attached seperately
