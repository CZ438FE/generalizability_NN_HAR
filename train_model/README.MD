## Train_Model

Provides all the necessary utilities to train a model. It uses the pytorch framework for the training of the model.


### Required Arguments:

-train_d **training_data**: Specify the folder, in which the program searches for the trainings data 


### Optional Arguments:

-t **type**: Specify the type of Neural Network to train. Available Options are Feedforward Neural Networks (FFNN),
    Convolutional Neural Networks (CNN) and Recurrent neural Networks (RNN). One of [**FFNN**, **CNN**, **RNN**]. Default: **FFNN** 

-val_d **validation_data**: Specify the folder, in which the program searches for the validation data 

-hier_m **hierarchical_model**: When classifying hierarchical, featured data this parameter determines which classifier to train. 
    This determines the data available to the model. One of [**top**, **lifting**, **walking**]. Default: **top** 

-m **minibatch_size**: Specify the size of the minibatches with which to train, default: 64

-e **epochs**: 'Specify the amount of epochs with which to train, training for parts of epochs is possible. 
    When exploring hyperparameters, choosing a smaller subset of the available data as trainings-data might improve 
    calculation speed significantly, as loading this set into the RAM might be possible. default: **1.0**

-l **loss_function**: Specify the loss function to use, default: **CEL**

-op **loss_function**: Specify the optimizer to use, dafault: **adam**

-d **dropout_rate**: Specify the dropout-rate, default: **0.0**

-ls **layer_structure**: Specify the layer-structure of the NN to build by specifying the width(s) of the hidden layers.
    For Convolution layers the syntax is:  conv_<nr_filters>_<kernel_size>_<stride>
    For pooling layers the syntax is: <pooling-method>_pool_<kernel-size>_<stride> . valid pooling methods are [**max**, **mean**]
    
    For Recurrent Neural Networks the first given hidden layer determines the size of the self-connected layer, the following layers
    are therefore standard fully-connected feedforward-layers.
    
    Each layer is separated by f. default: **250f150f50**

-b **batch_normalization**: Set if you want to use batch_normalization, default: **False**

-lr **learning_rate**: Specify the initial learning rate to use, default: **0.001**'

-nr_m **number_minibatches**: Specify the number of minibatches to train

-p **plot**: Set if you want to see visualizations regarding change of loss and accuracy over the minibatches, default: **False**

-o **Output_date**: Specify the output_date, meaning in which folder the the model and the plots shall be saved, otherwise the current date and time are used, format YYYY-MM-DDThh:mm


If --dryrun (-d) is not given as argument for the main function, it saves the resulting windows in a folder  <data_path>/train_model/<type>/<output_date>


Regarding the correct order of preprocessing steps consult the README.md for the main.py . 
