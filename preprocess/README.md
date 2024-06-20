## Preprocess

Contains some general-purpose preprocessing utilities, which can be individually triggered by specifying a method:

**reduce_data** uses a given level of granularity to iterate over a set of files containing features and keeps only the columns, which are useful for a human activity recognition analyzer at the given granularity level. The main purpose of this method is therefore data-reduction via removing the features, which are not being used for the baseline model at the given granularity. Therefore reduce_data is only used for data in feature-format.

**balancing_over** implements a stable way to oversample the data for the granularity level, returns a collection of files with several windows and the respective data with balanced label distribution at the given granularity. All data for the training or evaluation of neural networks need to be passed through this functionality.

**prepare_dataset** prepares oversampled data for the training of the model. Therefore the data of each window is being saved as an individual file. The label gets prepared and saved together with the respective filename in the labels.csv. All data for the training or evaluation of neural networks need to be passed through this functionality.

This chapter expects the existence of a anonymization_file.json within the datapath directory containing information about the features needed for reduce_data.


### Required Arguments:

-i **Input_data**: Specify the absolute path to the input folder, in which the program searches for the data to handle. 

-m **Method**: Specify the method with which preprocess the data. One of [**reduce_data**, **balancing_over**, **balancing_evaluation**, **prepare_dataset**]


### Optional Arguments:

-g **Granularity**: Specify the label-granularity with which to handle the data. One of [**top**, **mid**], Tells the program, which features shall be discarded by **reduce_data**, how the labels shall be counted before oversampling for balancing and how the label is to be preprocessed by **prepare_dataset**. Default: **mid** 

-c **grid**: Choose this parameter in combination with the method prepare_dataset to save the resulting data in a format ready for the training of a CNN or RNN. Setting this parameter breaks the one-dimensional vector of one sample into a matrix of format [nr_channels, nr_timepoints], Default: False

-p **plot**: Set if you want to see visualizations regarding class distributions. Default: False

-o **Output_date**: Specify the output_date, meaning in which folder the resulting files shall be saved, otherwise the current date and time are used, format YYYY-MM-DDThh:mm


If --dryrun (-d) is not given as argument for the main function, it saves the resulting windows in a folder  <data_path>/preprocess/<method>/<output_date>


Regarding the correct order of preprocessing steps consult the README.md for the main.py . 
