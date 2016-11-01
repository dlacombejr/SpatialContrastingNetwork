# Spatial Contrasting Network in Keras

Implementation of [Spatial Contrasting Network](https://arxiv.org/abs/1610.00243) in [Keras](https://keras.io/). Included are three models: 

* Baseline Convolutional Neural Network
	.* Uses the same architecture reported in paper
	.* Able to replicate results of ~73% accuracy when using 4000 training examples

* Spatial Contrasting Network
	.* Trained to embed patches from same image closer in deep space than patches from other images

* Pre-trained Convolutional Neural Network
	.* Same architecture as baseline conv-net except early layers are initialized with weights learned from Spatial Contrasting Network
	.* Performance falls short of baseline conv-net

Images of each network and the weights from their first layers are in `/saved` (`/models` and `/images` sub-directories, respectively). 

Please help me reproduce the results from the paper! 
