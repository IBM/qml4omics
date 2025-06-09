import os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


#####


class QuantumSage():

    '''
    Sage class that will run an ML model over the input data frame which would be some set of defined data characeristics
    and performance metrics associated to the dataset the method use.  Right now it is focused on learning from just the data
    characteristics but it can eventual also include the model parameters as part of the input
    '''

    def __init__(self, data_input):
        '''
        This function initializes the Sage with the input data frame that contains the data characteristics and performance metrics
        '''

        self._columns_data_features = [ '# Features', '# Samples',
                                        'Feature_Samples_ratio', 'Intrinsic_Dimension', 'Condition number',
                                        'Fisher Discriminant Ratio', 'Total Correlations', 'Mutual information',
                                        '# Non-zero entries', '# Low variance features', 'Variation', 'std_var',
                                        'Coefficient of Variation %', 'std_co_of_v', 'Skewness', 'std_skew',
                                        'Kurtosis', 'std_kurt', 'Mean Log Kernel Density',
                                        'Isomap Reconstruction Error', 'Fractal dimension', 'Entropy',
                                        'std_entropy']
        self._columns_metrics = ['accuracy', 'f1_score', 'auc']
        self._columns_metadata = ['Dataset', 'embeddings','datatype', 'model_embed_datatype', 'iteration', 'model']

        self._input_data_features_only = data_input[self._columns_data_features]
        self._input_data_metrics = data_input[self._columns_metrics]
        self._input_data_metadata = data_input[self._columns_metadata]

        self._available_models = list(set(self._input_data_metadata['model']))
        if 'none' in self._available_models:
            self._available_models.remove('none')
        self._available_models.sort()
        self._available_embeddings = list(set(self._input_data_metadata['embeddings']))
        self._available_embeddings.sort()
        self._available_metrics = self._columns_metrics
        self._available_metrics.sort()

        self._results_subsages = {}

        self.set_seed()


    # TODO: trained sage should predict over every metric so that the user can decide what they want predicted
    def predict(self, input_data, metric = 'f1_score'):
        '''
        This function is used to make the prediction for a given metric on each of the models.
        The input data should be a DataFrame with the same features as the training data.
        The metric should be one of the available metrics (f1_score, auc, accuracy).
        It returns a DataFrame with the predictions for each model and the R2 score of the prediction.
        The predictions are sorted by the product of the metric and the R2 score, so that the best performing model is at the top.
        If the metric is not one of the available metrics, it will return None.
        If the input data does not have the same features as the training data, it will raise an error.

        Args:
            input_data (pd.DataFrame): DataFrame with the same features as the training data.
            metric (str): The metric to predict. Should be one of the available metrics (f1_score, auc, accuracy).

        Returns:
            predictions_df (pd.DataFrame): DataFrame with the predictions for each model and the R2 score of the prediction.
        '''

        predictions = []
        for model in self._available_models:
            pred = self._results_subsages[metric][model]['fit_model'].predict(input_data)[0]
            r2 = self._results_subsages[metric][model]['r2']
            predictions.append([model, pred, r2])
        predictions_df = pd.DataFrame( predictions, columns = ['model',metric,'r2'] )
        predictions_df[metric+'*r2'] = predictions_df[metric] * predictions_df['r2']
        predictions_df = predictions_df.sort_values(metric+'*r2', ascending=False)
        return predictions_df


    def train_sub_sages(self, test_size=0.2, sage_type = 'random_forest'):

        '''
        This function trains the actual sage for each ML method that was tested previously.
        It will train a sub-sage for each metric and each model.  The sub-sage will be trained on the features of the input data
        and the corresponding metric for each model.  The sub-sage will be trained on a train-test split of the data, using either a random forest 
        or a MLP regressor depending on the sage_type parameter.  The results of the training will be stored in the _results_subsages dictionary, which will have the following structure:
        _results_subsages = {
            'metric1': {
                'model1': {
                    'fit_model': <trained model>,
                    'preds': <predictions on test set>,
                    'y_test': <true values on test set>,
                    'params': <model parameters>,
                    'mae': <mean absolute error>,
                    'mse': <mean squared error>,
                    'rmse': <root mean squared error>,
                    'r2': <R2 score>
                },
                ...
            },
            ...
        }

        This function will iterate over each metric and each model, and train the corresponding sub-sage, while printing the progress.

        Args:
            test_size (float): The proportion of the data to be used for the test set. Default is 0.2.
            sage_type (str): The type of sage to be used. Can be 'random_forest' or 'mlp'. Default is 'random_forest'.

        Returns:
            None: The function does not return anything, it just trains the sub-sages and stores the results in the _results_subsages dictionary.  
        '''
        for metric in self._available_metrics:
            print(f"Working on {metric}")

            self._results_subsages[metric] = {}
            for model in self._available_models:
                model_indices = self._input_data_metadata[ self._input_data_metadata['model'] == model ].index
                X = self._input_data_features_only.loc[ model_indices ]
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                y = self._input_data_metrics.loc[model_indices][metric].fillna(0).to_numpy()
                
                print(f"Working on {model}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = self._seed)

                # TODO: include the parameters that may be part of training or learning which parameters are best

                if sage_type == 'random_forest':
                    self._results_subsages[metric][model] = self._sage_random_forest(X_train, X_test, y_train, y_test)
                elif sage_type == 'mlp':
                    self._results_subsages[metric][model] = self._sage_mlp(X_train, X_test, y_train, y_test)
                else:
                    return None

    def _sage_mlp(self, X_train, X_test, y_train, y_test, n_iter = 50):

        '''
        This function performs a randomized search of the parameter space in order to find the best settings for the MLP
        and then make a final prediction on the test set.  The randomize search is doing a 5 fold cross validation.
        The MLPRegressor is trained with the best parameters found in the randomized search.
        The function returns a dictionary containing the trained model, predictions, true values, parameters, and evaluation metrics.
        The evaluation metrics include mean absolute error (mae), mean squared error (mse), root mean squared error (rmse), and R2 score (r2).
        This function is used to train a sub-sage for the MLP model.
        It is called by the train_sub_sages function and is not meant to be called directly by the user, and is designed to be used with the 
        input data that has been preprocessed and split into training and test sets.
        
        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Test labels.
            n_iter (int): Number of iterations for the randomized search. Default is 50.

        Returns:
            result (dict): A dictionary containing the trained model, predictions, true values, parameters, and evaluation metrics.
        '''

        param_distributions = {"hidden_layer_sizes": [1,50], 
                               "activation": ["identity", 
                               "logistic", "tanh", "relu"], 
                               "solver": ["lbfgs", "sgd", "adam"], 
                               "alpha": [0.00005,0.0005]}

        #TODO: Parameter optimization

        model = MLPRegressor(hidden_layer_sizes=(32,10), 
                             activation='relu', 
                             solver='adam',
                             alpha=0, 
                             batch_size='auto',
                             learning_rate='adaptive', 
                             learning_rate_init=0.001, 
                             max_iter=1000, 
                             random_state=self._seed, 
                             n_iter_no_change=10)
        
        # Train
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        params = model.get_params()

        # Evaluate on held out
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        result = {
            'fit_model' : model,
            'preds' : preds,
            'y_test' : y_test,
            'params' : params,
            'mae' : mae,
            'mse' : mse,
            'rmse' : rmse,
            'r2' : r2
        }

        return result


    def _sage_random_forest(self, X_train, X_test, y_train, y_test, n_iter = 50):

        '''
        This function performs a randomized search of the parameter space in order to find the best settings for the Random Forest
        and then make a final prediction on the test set.  The randomized search is doing a 5 fold cross validation.
        The RandomForestRegressor is trained with the best parameters found in the randomized search.
        The function returns a dictionary containing the trained model, predictions, true values, parameters, and evaluation metrics.
        The evaluation metrics include mean absolute error (mae), mean squared error (mse), root mean squared error (rmse), and R2 score (r2).
        This function is used to train a sub-sage for the Random Forest model.  
        It is called by the train_sub_sages function and is not meant to be called directly by the user, and is designed to be used with the
        input data that has been preprocessed and split into training and test sets.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Test labels.
            n_iter (int): Number of iterations for the randomized search. Default is 50.
        Returns:
            result (dict): A dictionary containing the trained model, predictions, true values, parameters, and evaluation metrics.
        '''

        param_distributions = {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(5, 20),
            'min_samples_split': np.arange(2, 10),
            'min_samples_leaf': np.arange(1, 5),
            'bootstrap': [True, False]
        }

        # Initialize the Random Forest Classifier
        rf = RandomForestRegressor(random_state=self._seed)

        # Initialize RandomizedSearchCV
        rf_random = RandomizedSearchCV(estimator=rf, 
                                    param_distributions=param_distributions, 
                                    n_iter=n_iter, 
                                    cv=5, 
                                    random_state=self._seed,
                                    n_jobs=-1)
        
        # Train
        rf_random.fit(X_train, y_train)
        preds = rf_random.predict(X_test)
        params = rf_random.best_params_

        # Evaluate on held out
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        result = {
            'fit_model' : rf_random,
            'preds' : preds,
            'y_test' : y_test,
            'params' : params,
            'mae' : mae,
            'mse' : mse,
            'rmse' : rmse,
            'r2' : r2
        }

        return result

    def plot_results(self, saveFile='' ):

        ''' This function plots the results of the sub-sages trained on the input data.
        It will create a bar plot for each metric showing the performance of each model, and a scatter plot of the predictions vs. true values.
        The bar plot will show the mean absolute error (mae), mean squared error (mse), root mean squared error (rmse), and R2 score (r2) for each model.
        The scatter plot will show the predictions vs. true values for each model.
        If saveFile is provided, the plots will be saved to that file. Otherwise, the plots will be shown.
        It is designed to be used after the train_sub_sages function has been called, and the sub-sages have been trained.
        Args:
            saveFile (str): The file name to save the plots. If empty, the plots will be shown. Default is ''.
        Returns:
            None: The function does not return anything, it just plots the results of the sub-sages.
        
        '''

        results = []
        preds = pd.DataFrame()
        for metric in self._available_metrics:
            for model in self._available_models:
                scores = pd.Series( self._results_subsages[metric][model].values(), index=self._results_subsages[metric][model].keys())
                results.append( [model, metric]+list(scores[['mae','mse','rmse','r2']]) )
                p = self._results_subsages[metric][model]['preds']
                y = self._results_subsages[metric][model]['y_test']
                preds = pd.concat( [preds,
                                    pd.DataFrame( [[model]*len(p),[metric]*len(p),p,y], index = ['model', 'metric', 'pred', 'y_test'] ).transpose() ] )
            results_df = pd.DataFrame(results, columns=['model','metric','mae','mse','rmse','r2'])

        results_df = results_df.melt(id_vars=['model', 'metric'])
        for metric in self._available_metrics:
            plt.figure(figsize=(6,4))
            sns.barplot(data = results_df[results_df['metric']==metric], x = 'variable', y = 'value', hue = 'model', hue_order=self._available_models)
            plt.title( "Predictive performance for each model for " + metric)
            plt.tight_layout()
            if saveFile != '':
                plt.savefig(saveFile)
            plt.show()
            plt.close()

            toPlot = preds[ preds['metric'] == metric ]
            plt.figure(figsize=(6,6))
            plt.title( "Predictive performance for each model for " + metric)
            sns.scatterplot( data = toPlot, x = 'y_test', y = 'pred', hue = 'model' )
            plt.tight_layout()
            plt.show()
            plt.close()


    def set_seed(self, seed=42):
        self._seed = seed

