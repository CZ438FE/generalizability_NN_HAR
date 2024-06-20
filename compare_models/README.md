## Compare_models 

Is applied to evaluated models to perform statistical tests on their classification behavior. All implemented statistical tests assume that the both models were evaluated on the same data.

As input this service expects the full absolute path to two folders containing the results from model evaluation. This service can also be applied to models that were gathered via the evaluation of several hierarchical classifiers, i.e. the folders used as input may result from calling the service evaluate_hierarchical. 


The available statistical tests are:
- McNemars Test
- Wilcoxon signed-rank Test
- Friedman Test


### Required Arguments:

-m1 **Folder of evaluated model 1**: Specify the absolute path to the folder with the evaluations of the first model to be compared

-m2 **Folder of evaluated model 2**: Specify the absolute path to the folder with the evaluations of the second model to be compared


### Optional Arguments:


-a **other_model_folders**: Specify the absolute path to other folders of evaluated models (needed only for the friedman test)

-a **Alpha**: Specify the alpha value to be used for the statistical test, i.e. the required significance-level needed to reject the respective null-hypotheses

-o **Output_date**: Specify the output_date, meaning in which folder the results shall be saved, otherwise the current date and time are used, format YYYY-MM-DDThh:mm



If --dryrun (-d) is not given as argument for the main function, it saves the resulting windows in a folder  <data_path>/compare_models/<output_date>


Regarding the correct order of preprocessing steps consult the README.md for the main.py . 