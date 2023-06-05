import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_and_save_distribution(data, filename):
    plt.figure()
    plt.hist(data, density=False, edgecolor='black', bins=50)

    plt.xlabel('Valeur de faithfulness')
    plt.ylabel('Nombre d\'images')
    plt.title('Distribution statistique')

    mean = np.mean(data)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)

    plt.savefig(filename)

def normalize_minmax(data):
    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        normalized_data = [0 for x in data]
    else:
        normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized_data

import matplotlib.pyplot as plt

def plot_histograms(data, filename='histogram.png'):

    plt.figure()

    for d in data:
        plt.hist(d[1], bins=50, density=False, alpha=0.6, edgecolor='black', label = d[0])

    plt.xlabel('Valeurs de faithfulness normalisées')
    plt.ylabel('Nombre d\'images')
    plt.title('Histogrammes normalisés')
    plt.legend()

    plt.savefig(filename)

def parser_method_dict(df, batch = 16):
    dataf = pd.DataFrame(columns = [f"{i}" for i in range(batch)])
    for i in range(df.shape[0]):
        row = yaml.safe_load(df.iloc[i].iloc[0])
        dataf.loc[i] = [np.trapz(row[j]) for j in row]
    return dataf

def parser_method_dict_with_layers(df, batch = 1):
    dataf = pd.DataFrame(columns = [f"{i}" for i in range(batch)])
    for i in tqdm(range(df.shape[0])): # loop over the number of batches

        row = yaml.safe_load(df.iloc[i].iloc[0]) # Get the batch result

        p_list = []
        for p in range(batch):
            j_list = []
            for j in row:
                j_list.append(row[j][p])
            for x in range(len(j_list)):
                if j_list[x] == 'nan':
                    j_list[x] = 0
            p_list.append(np.trapz(j_list))
        dataf.loc[i] = p_list
 
    return dataf

'''transform = {'Monotonicity Nguyen': lambda x: x, 
             'Local Lipschitz Estimate': lambda x: -x, 
             'Faithfulness Estimate': abs, 
             'Faithfulness Correlation': abs, 
             'Avg-Sensitivity': lambda x: -x, 
             'Random Logit': lambda x: x,
             'Sparseness': lambda x: x, 
             'EffectiveComplexity': lambda x: -x,
             'Nonsensitivity': lambda x: -x, 
             'Pixel-Flipping': lambda x: x.apply(lambda row: - np.trapz(row), axis=1),
             'Max-Sensitivity': lambda x: -x, 
             'Complexity': lambda x: -x, 
             "Selectivity": lambda x: -parser_method_dict(x, batch=1), 
             'Model Parameter Randomisation': lambda x: parser_method_dict_with_layers(x),
             'Monotonicity Arya': lambda x: x,
            }'''

transform = {'Monotonicity Nguyen': lambda x: x, 
             'Local Lipschitz Estimate': lambda x: x, 
             'Faithfulness Estimate': abs, 
             'Faithfulness Correlation': abs, 
             'Avg-Sensitivity': lambda x: x, 
             'Random Logit': lambda x: x,
             'Sparseness': lambda x: x, 
             'EffectiveComplexity': lambda x: x,
             'Pixel-Flipping': lambda x: x.apply(lambda row: np.trapz(row), axis=1),
             'Max-Sensitivity': lambda x: x, 
             'Complexity': lambda x: x, 
             "Selectivity": lambda x: parser_method_dict(x, batch=1), 
             'Model Parameter Randomisation': lambda x: parser_method_dict_with_layers(x),
             'Monotonicity Arya': lambda x: x,
            }
    
metrics =  ['Model Parameter Randomisation']

methods = ['random']

if __name__ == '__main__':

    results = {}

    with open(f"result2.csv", 'w', newline='') as f:
        for metr in metrics:
            data = pd.DataFrame()
            print("-- Metric: ", metr)

            for meth in methods:

                csv_name = f"csv/2000idx/{meth}_vit_b16_imagenet_{metr}.csv"

                df = pd.read_csv(csv_name, header=None)
                arr_values = transform[metr](df).values.flatten()
                                
                norm_arr_values = np.array(normalize_minmax(arr_values))

                results[metr] = norm_arr_values

                mean_values = round(norm_arr_values.mean(axis=0), 3)

                res = f"{meth} {metr} : {mean_values}\n"
                f.write(res)
              

                

            
"""
    faithfulness_metrics = ['Faithfulness Correlation', 
                            'Faithfulness Estimate', 
                            'Pixel-Flipping', 
                            'Region Perturbation', 
                            'Monotonicity Arya',
                            'Monotonicity Nguyen',
                            'Selectivity',
                            'SensitivityN',
                            'IROF']
    
    data_faithfulness = [(k, v) for k, v in results.items() if k in faithfulness_metrics]

    plot_histograms(data_faithfulness, filename='results/faithfulness_histogram.png')
"""