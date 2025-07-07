
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compute_results_correlation( results_df, correlation = 'spearman', thresh = 0.7 ):
    
    """This function takes in as input a Pandas Dataframe containing the results and data evaluations for
    a given dataset.  It then produces a spearman correlation between the data evaluation characteristics (features)
    and instances where an F1 score was observed above a certain threshold (thresh).
    The function returns the input DataFrame with additional columns for datatype and model_embed_datatype,
    as well as a new DataFrame containing the computed correlations between metrics and features.
    The correlation is computed for each model-embedding-dataset combination, and the results are aggregated.
    The features considered for correlation include various data characteristics such as 'Feature_Samples_ratio', 'Intrinsic_Dimension', etc.
    The metrics considered for correlation include 'accuracy', 'f1_score', 'time', and 'auc'.
    The function also calculates the median metric value and the fraction of instances above the specified threshold for each combination.
    The resulting DataFrame contains the model-embedding-dataset, metric, feature, median metric value, fraction above threshold, and the computed correlation.
    This function is useful for understanding how different data characteristics relate to model performance metrics, particularly in the context of machine learning models applied to datasets.

    Args:
        results_df (pd.DataFrame): A DataFrame containing the results and data evaluations.
        correlation (str): The type of correlation to compute, default is 'spearman'.
        thresh (float): The threshold for F1 score to consider, default is 0.7.
    
    Returns:
        results_df (pd.DataFrame): The input DataFrame with additional columns for datatype and model_embed_datatype.
        correlations_df (pd.DataFrame): A DataFrame containing the computed correlations between metrics and features.

    """

    # Refining datasrame
    results_df['datatype'] = [ re.sub( '\.csv', '', re.sub( '-.*', '', x ) ) for x in results_df['Dataset'] ]
    results_df[ 'model_embed_datatype'] = [ '_'.join( [str(row.model), str(row.embeddings), str(row.datatype)] ) for idx, row in results_df.iterrows() ]

    correlations = []
    features = ['Feature_Samples_ratio', 'Intrinsic_Dimension', 'Condition number',
        'Fisher Discriminant Ratio', 'Total Correlations', 'Mutual information',
        '# Non-zero entries', '# Low variance features', 'Variation', 'std_var',
        'Coefficient of Variation %', 'std_co_of_v', 'Skewness', 'std_skew',
        'Kurtosis', 'std_kurt', 'Mean Log Kernel Density',
        'Isomap Reconstruction Error', 'Fractal dimension', 'Entropy',
        'std_entropy']
    metrics = ['accuracy', 'f1_score', 'time', 'auc']
    
    keys = list(set(results_df['model_embed_datatype'])) 
    for m in keys:
        dat_temp_m = results_df[results_df['model_embed_datatype'] == m]
        if len(dat_temp_m) > 0:
            for s in metrics:
                for f in features:
                    if f in dat_temp_m.columns:
                        if correlation == 'spearman': 
                            correlations.append( [m, s, f, np.median(dat_temp_m[s]), sum(dat_temp_m[s]>thresh)/len(dat_temp_m[s]), spearmanr( dat_temp_m[s], dat_temp_m[f] )[0] ] )
                        
    correlations_df = pd.DataFrame(correlations, columns = ['model_embed_datatype', 'metric', 'feature', 'median_metric', 'frac_gt_thresh', 'correlation'] )

    return results_df, correlations_df

def plot_results_correlation( correlations_df, metric = 'f1_score', title = '', correlation_type = '', figsize=(6,6), save_file_path = '', size = 'correlation'):
    
    """This function plots the spearman correlation dot plots using the previously generated correlations_df dataframe. 
    The larger the circle, the higher the F1 score for that particular data set. The circle colors correspond to the 
    correlations between the data characteristics (evaluations) and the F1 score. Red corresponds to a postive 
    correlation, while blue indicates an anti-correlation.  The strength of either type of correlation is represented by 
    the shade of coloring -- the darker the circle, the more correlated/anticorrelated that particular characteristic is
    to the model's performance. 

    Args:
        correlations_df (pd.DataFrame): A DataFrame containing the computed correlations between metrics and features.
        metric (str): The metric to plot, default is 'f1_score'.
        title (str): The title of the plot, default is an empty string.
        correlation_type (str): The type of correlation to display in the legend, default is an empty string.
        figsize (tuple): The size of the figure, default is (6, 6).
        save_file_path (str): The file path to save the plot, default is an empty string.
        size (str): The column name to use for the size of the dots, default is 'correlation'.
        
    Returns:
        None: Displays the plot and saves it to the specified file path if provided. 
    """
    
    # Use CenteredNorm to center the colormap at 0
    norm = mcolors.CenteredNorm(vcenter=0)

    # Sample data
    data = correlations_df[correlations_df['metric'] == metric]
    data = data.sort_values( 'feature' )
    data['datatype'] = [ '_'.join( x.split('_')[2:] ) for x in data['model_embed_datatype']]
    data = data.sort_values( 'datatype' )
    data['model'] = [ re.sub( '_.*', '', x ) for x in data['model_embed_datatype']]
    data = pd.concat( [data[ ~data['model'].isin( ['qsvc', 'qnn', 'vqc', 'pqk']) ], data[ data['model'].isin( ['qsvc', 'qnn', 'vqc', 'pqk']) ] ] )
    fm = dict(zip( list(set(data['feature'])), range(len(set(data['feature']))) ) )
    data['feature_map'] = [ fm[x] for x in data['feature']]
    data['norm_size'] = [float(np.round(x*100)) for x in MinMaxScaler().fit_transform(data[size].values.reshape(-1,1))]

    data = data.fillna(0)
    
    plt.figure(figsize=figsize)
    ax = plt.scatter(data['model_embed_datatype'], data['feature'], s=data['norm_size'], 
                     c=data['correlation'], cmap='vlag', norm=norm)
    plt.xlabel('Model_Embedding_Dataset')
    plt.ylabel('Data Feature')
    plt.title(title)
    plt.xticks(rotation = 90)
    handles, labels = ax.legend_elements(prop="colors", alpha=0.6)
    handles3, labels3 = ax.legend_elements(prop="sizes", alpha=0.6)
    smin = np.min(data[size])
    smax = np.max(data[size])
    srate = (smax-smin)/(10-1)
    labels3 = [ round(float(x),2) for x in np.arange( smin, smax, srate)] + [round(smax,2)]
    legend3 = plt.legend(handles+handles3, labels+labels3, title=correlation_type, bbox_to_anchor=(1.4, 1), loc='upper right')
    plt.tight_layout() 
    if save_file_path != '':
        plt.savefig(save_file_path, dpi=300)
    plt.show()
    plt.close()