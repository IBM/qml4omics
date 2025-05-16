## function to find datasets where QML methods did better than classical
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def qml_winner(results_df, rawevals_df, output_dir, tag):
    """Find data sets where QML was beneficial (higher F1 scores than CML) and create new .csv files
    with the relevant evaluation and performance for these specific datasets, for further analysis.

    Args:
        results_df (pandas.DataFrame): Dataset in pandas corresponding to 'ModelResults.csv'
        rawevals_df (pandas.DataFrame): Dataset in pandas corresponding to 'RawDataEvaluation.csv'
    Returns: 
        qml_winners (pandas.DataFrame): contais the input datasets for which at least one QML method
                                        performed better than CML. DataFrame contains the scores of all
                                        the methods. 
        winner_eval_score (pandas.DataFrame): contains the input datasets, their evaluation, and scores for the 
                                            specific qml method that yielded the best score.
    """
    
    # pass in the ML results
    df = results_df.copy()
    # pull in the raw evaluations
    rawevals = rawevals_df.copy()
    #first, compute mean across all splits 
    df_across_split= df.groupby(['Dataset', 'embeddings', 'model', 'Model_Parameters'])['f1_score'].mean().reset_index()
    #now, extract the best results per method across embedding and iteration
    df_best = df_across_split.groupby(['Dataset', 'model'])['f1_score'].max().reset_index()
    #df_best = df_across_split.groupby(['Dataset', 'model', 'Model_Parameters'])['f1_score'].max().reset_index()
    df_best.to_csv(( os.path.join( output_dir,  tag +'_best_across_split.csv')), index=False)
    # get summary accross all datasets
    df_best_model_mean = df_best.groupby('model')['f1_score'].mean()
    df_best_model_median = df_best.groupby('model')['f1_score'].median()
    df_best_model_max = df_best.groupby('model')['f1_score'].max()
    df_best_model_std = df_best.groupby('model')['f1_score'].std()
    df_best_permodel_summary = pd.concat([df_best_model_mean, df_best_model_median, df_best_model_max, df_best_model_std], axis=1)
    df_best_permodel_summary.columns = ['Mean_F1_Score', 'Median_F1_Score', 'Max_F1_Score', 'StandardDev_F1_Score']
    df_best_permodel_summary.to_csv(( os.path.join( output_dir,  tag +'_best_permodel_summary.csv')))
    print(df_best_permodel_summary)
    
    # extract the best results per dataset
    best_per_dataset = df_best.loc[df_best.groupby('Dataset')['f1_score'].idxmax()]
    # best_per_dataset = df_across_split.loc[df_across_split.groupby('Dataset')['f1_score'].idxmax()]
    # create list of qml methods
    qml_list = ['qsvc', 'qnn', 'vqc', 'pqk']
    qml_winner = df_best[df_best['Dataset'].isin(best_per_dataset[best_per_dataset['model'].isin(qml_list)]['Dataset'])]
    #qml_winner = df_across_split[df_across_split['Dataset'].isin(best_per_dataset[best_per_dataset['model'].isin(qml_list)]['Dataset'])]
    if not qml_winner.empty:
        bestmethod = qml_winner.groupby('Dataset')['f1_score'].idxmax()
        qc_method_and_score = qml_winner.loc[bestmethod]
        qml_winner.to_csv(( os.path.join( output_dir,  tag +'_qml_winners.csv')), index=False)
        dataset = list(qml_winner['Dataset'].unique())
        
        #######
        # now let's find the raw data evaluations for the qml winner data sets
        # this wil produce another csv file that contains scores, evaluation, and qml method
        # for these "qml winners".
        winner_evals = []
        for file in dataset:
            eval = rawevals.loc[rawevals['Dataset'] == file]
            print(eval)
            winner_evals.append(eval)
        winner_evals_df = pd.concat(winner_evals)
        winner_evals_df.to_csv(( os.path.join( output_dir,  tag +'_winner_evals.csv')), index=False)
        winner_scores_df = qc_method_and_score.iloc[:, -3:]
        winner_scores_df.to_csv(( os.path.join( output_dir,  tag +'_winner_score.csv')), index=False)
        #winner_eval_score = pd.concat([winner_evals_df, winner_scores_df], axis=1)
        #winner_eval_score.to_csv(( os.path.join( output_dir,  tag +'_winner_eval_score.csv')), index=False) # contains dataset, evaluation, qml method, and  average f1 score 
        #######
        
        # optional print statements
        print('*** The number of qml winners is', len(dataset))
        print('*** The qml winners are:', dataset)
        
        return qml_winner#, winner_eval_score
    
    else:
        print('*** QML methods were outperformed by CML methods in all datasets ***')
    
        return 


