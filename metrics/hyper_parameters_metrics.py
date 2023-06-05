import quantus

def get_hyper_param_eval(img_size=224, num_classes=1000):
    # cifar10 
    if img_size == 32:
        small_patch_size = 2
        big_patch_size = 4
        num_classes = 10
    # ImageNet
    elif img_size == 224:
        small_patch_size = 14
        big_patch_size = 28
        num_classes = 1000
    else: raise NotImplementedError('Metrics hyperparameters not defined for image size different from 32 (CIFAR10) and 224 (ImageNet).')

    hyper_param_eval = {
        #Faithfullness
        'Faithfulness Correlation': {
            'nr_runs': 10,
            'subset_size': img_size,
            'perturb_baseline': 'black',
            'perturb_func': quantus.baseline_replacement_by_indices,
            'similarity_func': quantus.correlation_pearson,
            'abs': False,
            'return_aggregate': False,
            'disable_warnings': True
        },
        'Faithfulness Estimate': {
            'similarity_func': quantus.correlation_pearson,
            'perturb_func': quantus.baseline_replacement_by_indices,
            'img_size': img_size,
            'features_in_step': img_size,
            'perturb_baseline': 'black',
            'pixels_in_step': big_patch_size,
            'disable_warnings': True
        },
        'Pixel-Flipping': {
            'features_in_step': img_size,
            'perturb_baseline': 'black',
            'perturb_func': quantus.baseline_replacement_by_indices,
            'disable_warnings': True
        },
        'Region Perturbation': {
            'patch_size': big_patch_size,
            'regions_evaluation': 100,
            'img_size': img_size,
            'random_order': False,
            'perturb_func': quantus.baseline_replacement_by_patch,
            'perturb_baseline': 'uniform',
            'disable_warnings': True
        },
        'Monotonicity Arya':  {
            'features_in_step': img_size,
            'perturb_baseline': 'black',
            'perturb_func': quantus.baseline_replacement_by_indices,
            'similarity_func': quantus.correlation_spearman,
            'disable_warnings': True
        },
        'Monotonicity Nguyen': {
            'nr_samples': 10,
            'features_in_step': img_size,
            'perturb_baseline': 'uniform',
            'perturb_func': quantus.baseline_replacement_by_indices,
            'similarity_func': quantus.correlation_spearman,
            'disable_warnings': True
        },
        'Selectivity':{
            'patch_size': small_patch_size,
            'perturb_func': quantus.baseline_replacement_by_patch,
            'perturb_baseline': 'black',
            'disable_warnings': True
        },
        'SensitivityN': {
            'features_in_step': big_patch_size,
            'n_max_percentage': 0.8,
            'img_size': img_size,
            'similarity_func': quantus.correlation_pearson,
            'perturb_func': quantus.baseline_replacement_by_indices,
            'perturb_baseline': 'uniform',
            'disable_warnings': True
        },
        'IROF': {
            'segmentation_method': 'slic',
            'perturb_baseline': 'mean',
            'perturb_func': quantus.baseline_replacement_by_indices,
            'disable_warnings': True
        },
        #Localisation
        'Top-K Intersection': {},
        'Relevance Mass Accuracy': {},
        'Relevance Mass Ranking': {},
        'Attribution Localisation':{},
        'AUC':{},
        #Randomisation
        'Model Parameter Randomisation':{
            'layer_order': 'top_down',
            'similarity_func': quantus.correlation_spearman,
            'normalize': True,
            'disable_warnings': True
        },
        'Random Logit':{
            'num_classes': num_classes,
            'similarity_func': quantus.ssim,
            'disable_warnings': True
        },
        #Robustness
        'Continuity Test':{
            'nr_patches': 4,
            'nr_steps': 10,
            'img_size': img_size,
            'perturb_baseline': 'black',
            'similarity_func': quantus.correlation_spearman,
            'perturb_func': quantus.translation_x_direction,
            'disable_warnings': True
        },
        'Local Lipschitz Estimate': {
            'nr_samples': 10,
            'perturb_std': 0.1,
            'perturb_mean': 0.1,
            'norm_numerator': quantus.distance_euclidean,
            'norm_denominator': quantus.distance_euclidean,
            'perturb_func': quantus.gaussian_noise,
            'similarity_func': quantus.lipschitz_constant,
            'disable_warnings': True
        },
        'Max-Sensitivity':{
            'nr_samples': 10,
            'perturb_radius': 0.2,
            'norm_numerator': quantus.fro_norm,
            'norm_denominator': quantus.fro_norm,
            'perturb_func': quantus.uniform_sampling,
            'similarity_func': quantus.difference,
            'disable_warnings': True
        },
        'Avg-Sensitivity':{
            'nr_samples': 10,
            'perturb_radius': 0.2,
            'norm_numerator': quantus.fro_norm,
            'norm_denominator': quantus.fro_norm,
            'perturb_func': quantus.uniform_sampling,
            'similarity_func': quantus.difference,
            'disable_warnings': True
        },
        #Complexity
        'Sparseness': {},
        'Complexity':{},
        'EffectiveComplexity':{
            'eps': 1e-5,
            'disable_warnings': True
        },
        #Axiomatic
        'Completeness':{
            'abs': False,
            'disable_warings': True,
        },
        'Nonsensitivity':{
            'abs': True,
            'eps': 1e-5,
            'n_samples': 10,
            'perturb_baseline': 'black',
            'perturb_func': quantus.baseline_replacement_by_indices,
            'disable_warnings': True
        }
    }
    
    return hyper_param_eval

metric_types = {
    'robustness': ['Local Lipschitz Estimate', 'Avg-Sensitivity', 'Max-Sensitivity', 'Continuity Test'],
    'randomisation': ['Model Parameter Randomisation',  'Random Logit']
}
