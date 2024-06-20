## Generate features

Uses the windowed, non-flattened data in the input folder to generate all necessary features for the windows and saves the files containing the features afterwards.

This chapter expects the existence of a anonymization_file.json within the datapath directory containing information about the features to create.

### Required Arguments:

-i **Input_date**: Specify the input time, in which the program searches for the data to handle. Expects format : YYYY-MM-DDThh:mm
    The whole path to the input_data is therefore <data_path>/windows/<input_date>


### Optional Arguments:

-o **Output_date**: Specify the output_date, meaning in which folder the windows with the features shall be saved, otherwise the current date and time are used, format YYYY-MM-DDThh:mm


If --dryrun (-d) is not given as argument for the main function, it saves the resulting windows in a folder  <data_path>/features/<output_date>


Regarding the correct order of preprocessing steps consult the README.md for the main.py . 
