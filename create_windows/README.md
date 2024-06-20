## Create_Windows 

Is used to cut the collection of labeled data into sliding windows of the given **window_size** (in ms) and creates a new window after a given **step_length** (in ms) has passed. Additionally it performs some basic logical checks and transforms the time column to UNIX time in ms. For data consistency purposes the data is also resampled, the timespan to resample may also be given via the **resampling_rate** (in ms) parameter. Before the resampling is applied the data is being filled via a specified filling **method** and afterwards flattening the data may be applied (convert data to one-dimensional array) via the **flatten** keyword. Flattenening is to be applied, when the data is not used to generate features.


It expects a data_path which contains a folder called labeled_data, wherein the labeled data needs to be. 
Additionally it only saves the window if there are real datapoints within the calculated window. 
A real datapoint is defined as any datapoint not created from simply filling/interpolating from the last to the next observed datapoint.

Caution has been taken to ensure that the resulting windows loose as little information as possible.

This chapter expects the existence of a anonymization_file.json within the datapath directory containing information about the needed column names, as well as the parameters for the standardization of said parameters.

### Required Arguments:


### Optional Arguments:

-w **Window_length**: Specify the window length in ms (needs to be greater than 0). Default: 1000

-s **Step_length**: Specify the step length in ms (needs to be greater than 0 and smaller than the window length).  Default: 500

-m **Method**: Specify the method with which to fill the missing entries from the labeled_data, as the datapoints might not be evenly spaced 
    (either forwardfilling ffill or linear interpolation linear), Default: ffill

-rs **Resampling_rate**: Specify the resampling_rate in ms (needs to be greater than 0, smaller than 200 and must be <= window_length). Defines the period to be 
    aggregated into a new, single timepoint after fillling. Default: 10

-f **Flatten**: Set if you want to flatten the resampled window into a one-dimensional vector (set, if the data is not being used to generate features)

-n **Normalize**: Set if you want to apply normalization to the data, Options: [None, max, z_normalize], Default: None

-o **Output_date**: Specify the output_date, meaning in which folder the created windows shall be saved, otherwise the current date and time are used, format YYYY-MM-DDThh:mm

If --dryrun (-d) is not given as argument for the main function, it saves the resulting windows in a folder  <data_path>/windows/<output_date>


Regarding the correct order of preprocessing steps consult the README.md for the main.py . 
