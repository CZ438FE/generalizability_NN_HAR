## Evaluate_classwise 

Is a service dedicated to deliver class-wise evaluations of a classifier. This is being done by breaking up the classification problem of a multclass-classification problem with k possible classes into k class vs rest classification problems and reporting resulting metrics and ROC curves.


This service can be used to evaluate hierarchically classifying models as well.



### Required Arguments:


-top **Top-level-classifier folder**: Specify the absolute path to the classifier to be evaluated. When hierarchical classification was done, this needs to be folder of the model on top of the classification hierarchy, i.e. the one classifying first


### Optional Arguments:


-lift **Lifting Classifier folder**: Specify the full absolute path to the evaluated lifting subclassifier (needed only for hierarchical models)

-walk **Walking Classifier folder**: Specify the full absolute path to the evaluated walking subclassifier (needed only for hierarchical models)

-o **Output_date**: Specify the output_date, meaning in which folder the created windows shall be saved, otherwise the current date and time are used, format YYYY-MM-DDThh:mm


If --dryrun (-d) is not given as argument for the main function, it saves the resulting windows in a folder  <data_path>/evaluate_classwise/<output_date>


Regarding the correct order of preprocessing steps consult the README.md for the main.py . 
