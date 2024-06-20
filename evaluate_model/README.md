## Evaluate_Model

Provides all the necessary utilities to evaluate a trained model on test data.


### Required Arguments:

-m **model**: Specify the path to a directory, in which the program searches for the trained model

-test_d **test_data**: Specify the folder, in which the program searches for the test_data for the final evaluation


### Optional Arguments:

-o **Output_date**: Specify the output_date, meaning the name of the directory in which the results of the evaluation shall be saved, otherwise the current date and time are used, format YYYY-MM-DDThh:mm


If --dryrun (-d) is not given as argument for the main function, it saves the resulting windows in a folder  <data_path>/evaluate_model/<type>/<output_date>


Regarding the correct order of preprocessing steps consult the README.md for the main.py . 
