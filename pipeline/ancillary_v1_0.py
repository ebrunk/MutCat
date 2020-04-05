from scipy import stats
from scipy.stats import pearsonr 
from numpy.random import random_sample, seed
import numbers
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle
import numpy as np
from random import seed
import pandas as pd
from copy import deepcopy
import ccal

RANDOM_SEED = 20121020

def sim_anneal_1D_MDS(distance_matrix, 
                      max_iter = 3000, 
                      sigma = 0.20, 
                      init_temp = 0.0075,
                      random_seed = RANDOM_SEED):
    
    """ 
    Multidimensional Scaling from n dimensions (distance matrix) onto 1D ring (theta angles: 0 -- 2 pi)
    using simmulated annealing and pearson correlation as utility function.
    
    Arguments:
        distance matrix (DataFrame): symmetric distance matrix
        max_iter (int): number of time iterations
        sigma (float): scale for random changes in theta (stdev of Gaussian used to generate candidate values)
        init_temp (float): initial temperature
        random_seed (int): random seed for random number generator
    Returns:
        cor_vs_t (Series): time series of correlations vs time
        thetha (Series): set of theta's final configuration
        proj_distance_matrix: (DataFrame), projected distance matrix
    """
    # Initialize
    seed(random_seed)
    N = distance_matrix.shape[0]
    theta = pd.Series(2*np.pi*np.random.random_sample((N,)), index = distance_matrix.index)
    cor_vs_t = pd.Series(0, index = range(max_iter))
    proj_distance_matrix = pd.DataFrame(0, index=distance_matrix.index, columns=distance_matrix.columns)

    # Compute initial correlation between projected and original distance matrices
    for i in range(N):
        for j in range(i + 1, N):
            d = abs(theta.iloc[i] - theta.iloc[j])
            if (d > np.pi):
                d = 2*np.pi - d
            proj_distance_matrix.iloc[i, j] = proj_distance_matrix.iloc[j, i] = d**2
    
    cor_before = pearsonr(distance_matrix.values.flatten(), proj_distance_matrix.values.flatten())[0]
    
    # Simulated Annealing iteration loop
    for t in range(max_iter):
 
        # lower temperature linearly at every step
        temp = init_temp*(1 - t/(max_iter + 1))
        if t % 500 == 0:
            print('time: {}/{} correlation: {}'.format(t, max_iter, cor_before))     
    
        # generate proposed new value for theta (Gaussian distribution centered at current value)
        theta_loc = np.random.choice(range(N))
        old_theta = theta.iloc[theta_loc]
        new_theta = np.random.normal(old_theta, sigma, 1)
        if new_theta < 0:
            new_theta = 2*np.pi + new_theta
        elif new_theta > 2*np.pi:
            new_theta = new_theta - 2*np.pi
        theta.iloc[theta_loc] = new_theta

        # compute correlation between projected and original distance matrices after changing theta
        for i in range(N):
            for j in range(i + 1, N):
                d = abs(theta.iloc[i] - theta.iloc[j])
                if (d > np.pi):
                    d = 2*np.pi - d
                proj_distance_matrix.iloc[i, j] = proj_distance_matrix.iloc[j, i] = d**2         
    
        cor_after = pearsonr(distance_matrix.values.flatten(), proj_distance_matrix.values.flatten())[0]
    
        # compute acceptance probability (notice increase of correlation produces prob > 1 guranteeeing acceptance)
        prob = np.exp((1/temp)*(cor_after - cor_before))
    
        # print('before: {} after: {} prob: {}'.format(cor_before, cor_after, prob))
        
        # compare uniform random number vs. prob
        if np.random.random_sample((1,)) > prob:
            # change accepted
            theta.iloc[theta_loc] = old_theta
            # print('move rejected')
            cor_vs_t.iloc[t] = cor_before
        else:
            # change rejected
            # print('move accepted')
            cor_vs_t.iloc[t] = cor_after
            cor_before = cor_after
    
    # retunr time series, final theta configuration and projected distance matrix
    return {'cor_vs_t': cor_vs_t, 'theta': theta, 'proj_distance_matrix': proj_distance_matrix}

def make_radial_infograph(target,
                     features,
                     matching_scores,
                     direction = 'positive',
                     features_to_plot = (10, 0, 0), 
                     features_to_label = 'all_being_plotted', 
                     title = ' ',
                     plot_IC_thicks= True,
                     FDR_thicks = [0.05, 0.25],
                     alpha = 0.5,
                     max_iter = 1000, 
                     sigma = 0.20, 
                     init_temp = 0.0075,
                     figsize = [14, 14],
                     title_size = 30,
                     target_size = 300,
                     target_label_size = 30,
                     target_color = '#FF5555',
                     labels_color = '#4444FF',
                     labels_size = 12,
                     labels_delta = 0.02,
                     feature_size = 200,
                     features_color = '#6666EE',
                     features_to_label_color = '#66FF66',
                     rotation = np.pi/2,
                     file_path = None,
                     min_score = 'auto',
                     colors = None,
                     lines_to_center = False, 
                     thres_lines_to_center = 0.4,
                     random_seed=RANDOM_SEED):

    """ 
    Produce radial infograph to display top hits in a feature selection or matching analysis. This function
    should be called after calling make_mach_panel as it requires the output of that function as one of the inputs.
    This graph provides a visualization of the results of Differential Analysis using Mutual Information. 
    In this plot, the radial distance corresponds to the mutual information between each feature and the target, 
    and the radial distance is proportional to the inverse of the mutual information across features. In this way, 
    features that are associated with a given phenotype, and are also concordant on a sample by sample basis 
    will cluster and appear in the same sector of the plot. Alternatively, features that are also associated 
    with the target, but display significant differences on a sample by sample basis, or even display a 
    complementarity pattern will be displayed in opposite sectors. The development of this plot uses the 
    sim_anneal_1D_MDS function that maps a high-dimensionality distance matrix (based on the mutual information) 
    to a one-dimensional embedded dimension with circular geometry. 
    
    Arguments:
        target (Series): target (same as used in make_match_panel)
        features (DataFrame): features dataset (same as used in make_match_panel)
        matching_scores (DataFrame): output of make_match_panel
        direction (str): positive | negative, direction of matching
        features_to_plot (list or str): features to plot: [top, mid, bottom], or 'all' or feature names
        features_to_label (str or list)= 'all_being_plotted' or subset
        title (str): title of the plot
        plot_IC_thicks= True/False,
        FDR_thicks = [0.05, 0.25]
        alpha (float): power to rescale radial distances and provide zooming (< 1 zoon in, > 1 zoom out)
        max_iter (int): maximum number of simulated annealing iterations, 
        sigma (float): scale of simulated annealing changes for theta
        init_temp (float): initial simulatd annealing temperature
        figsize (list): figsize [x size, y size]
        title_size (int): title font size
        target_size (int): target symbol size
        target_label_size (int): target label font size
        target_color (hex/color): color for target symbol
        labels_color (hex/color): color for features symbol
        labels_size (int): feature labels font size
        feature_size (int): features symbol size
        features_color (hex/color): features symbol color
        features_to_label_color (hex, color): symbol color for features_to_label subset
        rotation (float): rotation for the plot in radiant units 
        file_path (str): file path to save the plot
        min_score ('auto' | number): minimum score to plot (auto selects min from data)
        colors (Series): A series of symbol colors for each feature and feature label
        lines_to_center: True/False plot lines from each feature to the center
        random_seed (int): random number generator seed
        
    Returns:
        MD_res (dictionary):
            cor_vs_t (Series): time series of correlations vs time
            thetha (Series): set of theta's final configuration
            proj_distance_matrix (DataFrame): projected distance matrix
    """

    seed(random_seed)

    overlap = set(target.index).intersection(features.columns)
    
    n_tot = matching_scores.shape[0]

    if isinstance(features_to_plot[0], numbers.Number):
        up_genes = features_to_plot[0]
        mid_genes = features_to_plot[1]
        dn_genes = features_to_plot[2]
        up = list(range(0, up_genes))
        mid = list(np.random.choice(range(up_genes, n_tot - dn_genes), size=mid_genes, replace=False))
        dn = list(range(n_tot - dn_genes, n_tot))
        up_mid_dn = up + mid + dn
        relevant_genes = matching_scores.iloc[up_mid_dn,:].index

    elif features_to_plot == 'all':
        relevant_genes = matching_scores.index

    else: 
        relevant_genes = set(matching_scores.index).intersection(features_to_plot)
    
    features = features.loc[relevant_genes, overlap]
    all_scores = matching_scores.loc[:, 'Score']
    scores = matching_scores.loc[relevant_genes, 'Score']
    if direction == 'negative':
        scores = - scores
        all_scores = - all_scores
    p_vals = matching_scores.loc[relevant_genes, 'P-Value']
    FDRs = matching_scores.loc[relevant_genes, 'FDR']
    all_FDRs = matching_scores.loc[:, 'FDR']  

    loc = np.argmin(abs(all_FDRs - 0.50))
    iloc = all_scores.index.get_loc(loc)
    all_FDRs.iloc[iloc:all_FDRs.size] = 1  
       
    distance_matrix = pd.DataFrame(0, index=features.index, columns=features.index)

    for i in features.index:
        for j in features.index:
            # print('i={}, j={}'.format(i,j))
            x = features.loc[i,:]
            y = features.loc[j,:]
            nans = np.isnan(x) | np.isnan(y)
            x = x[~nans]
            y = y[~nans]
            distance_matrix.loc[i, j] = ccal.compute_information_distance(x, y)
    
    distance_matrix2 =  1 - ((1/(1+np.exp(-alpha*distance_matrix))))
    
    MDS_res = sim_anneal_1D_MDS(distance_matrix, max_iter = max_iter, sigma = sigma, init_temp = init_temp)
    plt.plot(MDS_res['cor_vs_t'])

    # rotate plot
    theta = MDS_res['theta']
    theta = theta + rotation
   
    for i in range(len(theta)):
        if theta.iloc[i] > 2*np.pi:
            theta.iloc[i] = theta.iloc[i] - 2*np.pi

    radius = 1 - scores
    radius = radius**alpha
    
    if min_score == 'auto':
        min_score = all_scores.min()

    max_radius = (1 - min_score)**alpha    
              
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, aspect='equal', facecolor='w')

    plt.title(title, fontdict = {'fontsize': title_size, 'fontweight': 3})

    ax.set_xlim(-1.1*max_radius, 1.1*max_radius)
    ax.set_ylim(-1.1*max_radius, 1.1*max_radius)
    
    for i in range(len(x)):
        if colors is None or not scores.index[i] in colors.index:
            color = features_color
        else:
            color = colors.loc[scores.index[i]]
        ax.scatter(x[i], y[i], s = feature_size, c = color, marker = 'o', alpha=1) 
        if lines_to_center == True and scores[i] >= thres_lines_to_center:
            xx = [0, x[i]]
            yy = [0, y[i]]
            plt.plot(xx, yy, linewidth=1, color='#000000', linestyle='solid')
        
    ax.scatter(0, 0, s = target_size, c =target_color, marker = 'o', alpha=1) 
    ax.text(0.02, 0.02, target.name, horizontalalignment='left', verticalalignment='bottom', 
            fontsize=target_label_size)
    
    angles = np.arange(0, 2 * np.pi, 0.01)

    thick_num = 1
    for FDR in FDR_thicks:
        r = (1 - all_scores.loc[np.argmin(abs(all_FDRs - FDR))])**alpha
        angle_lab = thick_num * np.pi/16
        if thick_num % 2 == 1:
            align = 'right'
        else:
            align = 'left'
        plt.plot(r * np.cos(angles), r * np.sin(angles), linewidth=1, color='#DD0000', linestyle='dashed')
        ax.text(r * np.cos(angle_lab), r * np.sin(angle_lab), 'FDR: {} '.format(FDR), 
                horizontalalignment=align, color='#DD0000', verticalalignment='center', fontsize=10)    
        thick_num = thick_num + 1
        
    if plot_IC_thicks:
        for i in np.round(np.arange(min_score, 1, 0.1), decimals=1):
            if (i == 0):
                linestyle='dashed'
            else:
                linestyle='solid'
            plt.plot((1-i)**alpha * np.cos(angles), (1-i)**alpha * np.sin(angles), linestyle=linestyle, linewidth=1, 
                     color='#7dc4ff')
            ax.text(0, (1-i)**alpha + 0.01, i, horizontalalignment='center', color='#7dc4ff', verticalalignment='bottom', 
                     fontsize=10)    

    # plot external boundary    
    plt.plot(1.05*max_radius * np.cos(angles), 1.05*max_radius * np.sin(angles), linewidth=4, color='#9999FF')

    ax.set_axis_off()

    # plot labels
    if features_to_label == 'all_being_plotted':
        for i in range(len(x)):
            if colors is None or not scores.index[i] in colors.index:
                color = labels_color
            else:
                color = colors.loc[scores.index[i]]
                        
            ax.text(x[i]+labels_delta, y[i]-labels_delta, '{}'. format(scores.index[i]), horizontalalignment='left', verticalalignment='bottom', 
                fontsize=labels_size, color=color)
    else:
        for i in range(len(x)):
            if scores.index[i] in features_to_label:
                if colors is None or not scores.index[i] in colors.index:
                    color = features_to_label_color
                else:
                    color = colors.loc[scores.index[i]]
    
                ax.scatter(x[i], y[i], s = feature_size, c = color, marker = 'o', alpha=1, edgecolors = '#000000')
                ax.text(x[i]+labels_delta, y[i]-labels_delta, '{}'. format(scores.index[i]), horizontalalignment='left', verticalalignment='bottom', 
                    fontsize=labels_size, color=labels_color)

    plt.show()
    if file_path != None:
        fig.savefig(fname = file_path, dpi=100, bbox_inches='tight')
        
    return MDS_res


def produce_mutation_file(
    maf_input_file,
    gct_output_file,
    protein_change_identifier='Protein_Change',
    underscore_and_truncate_sample_names=False,
    variant_thres=80,
    change_thres=80,
    genes_with_all_entries=['NFE2L2', 'KEAP1'],
    only_select_from_list=False):
    
    print("Reading file...")
    d=pd.read_csv(maf_input_file, sep='\t', header=0, index_col=None, dtype=str)
    ds=d.loc[:,['Hugo_Symbol','Variant_Classification', 'Tumor_Sample_Barcode', protein_change_identifier]]

    sample_set=set()
    for i in ds['Tumor_Sample_Barcode']:
        sample_set.add(i)
    sample_dict=dict(zip(list(sample_set), range(0, len(sample_set))))

    if only_select_from_list:
        print("Selecting from list of genes...")
        final_dict={}
        for gene in genes_with_all_entries:
            gene_ds=ds.loc[ds['Hugo_Symbol']==gene]
            for i in gene_ds.index:
                sample_key=sample_dict[gene_ds.loc[i, 'Tumor_Sample_Barcode']]

                protein_change=gene_ds.loc[i, protein_change_identifier]
                variant_class=gene_ds.loc[i, 'Variant_Classification']

                if not protein_change is np.nan: 
                    if not gene+'_'+protein_change in final_dict:
                        final_dict[gene+'_'+protein_change] = [0]*len(sample_dict)
                    final_dict[gene+'_'+protein_change][sample_key]=1
                
                if variant_class != 'Silent': 
                    if not gene+'_Nonsilent' in final_dict:
                        final_dict[gene+'_Nonsilent'] = [0]*len(sample_dict)
                    final_dict[gene+'_Nonsilent'][sample_key]=1
        
                if not gene+'_'+variant_class in final_dict:
                    final_dict[gene+"_"+variant_class] = [0]*len(sample_dict)
                final_dict[gene+'_'+variant_class][sample_key]=1            
         
                if not gene+'_MUT_All' in final_dict:
                    final_dict[gene+"_MUT_All"] = [0]*len(sample_dict)
                final_dict[gene+'_MUT_All'][sample_key]=1            
    else:

        print("Counting each type of variant...")
        #Keep track of how many variants per gene
        gene_variant_count={}

        for i in ds.index:
            symbol=ds.loc[i,'Hugo_Symbol']
        
            if not ds.loc[i, protein_change_identifier] is np.nan:
                variant_name=ds.loc[i, protein_change_identifier]
        
        
            if symbol in gene_variant_count:
            
                gene_variant_count[symbol]['variant'].add(variant_name)
            
                if ds.loc[i, 'Variant_Classification'] not in gene_variant_count[symbol]:
                
                    gene_variant_count[symbol][ds.loc[i,'Variant_Classification']]=1
                
                    gene_variant_count[symbol]['MUT_All']=1

                    if ds.loc[i, 'Variant_Classification']!='Silent' and ('Nonsilent' not in gene_variant_count[symbol]):
                        gene_variant_count[symbol]['Nonsilent']=1
                    
                else:
                    
                    if ds.loc[i, 'Variant_Classification']!='Silent':
                        gene_variant_count[symbol]['Nonsilent']+=1
                    
                    gene_variant_count[symbol]['MUT_All']+=1
                    gene_variant_count[symbol][ds.loc[i,'Variant_Classification']]+=1
        
            else:
                gene_variant_count[symbol]={}
                gene_variant_count[symbol]['variant']=set([variant_name])
                if ds.loc[i, 'Variant_Classification']!='Silent':
                    gene_variant_count[symbol]['Nonsilent']=1
                gene_variant_count[symbol][ds.loc[i, 'Variant_Classification']]= 1
                gene_variant_count[symbol]['MUT_All']=1


        if genes_with_all_entries==None:
            genes_with_all_entries=[]
        
        _gene_variant_count=deepcopy(gene_variant_count)
        
        print("Dropping variants under threshold...")
        for gene in _gene_variant_count:
            if not (gene in genes_with_all_entries):
                for variant_type in _gene_variant_count[gene]:       
                    if variant_type == 'variant':
                        if len(_gene_variant_count[gene]['variant'])<change_thres:
                        
                            gene_variant_count[gene].pop('variant')
                            if len(gene_variant_count[gene].values())==0:
                                gene_variant_count.pop(gene)
                    else:
                        if _gene_variant_count[gene][variant_type]<variant_thres:
                            gene_variant_count[gene].pop(variant_type)
                            if len(gene_variant_count[gene].values())==0:
                                gene_variant_count.pop(gene)

        final_dict={}

        print("Creating final table...")
        #Go through table line by line
        for gene in gene_variant_count:
            gene_ds=ds.loc[ds['Hugo_Symbol']==gene]
            for i in gene_ds.index:
                variant_class=gene_ds.loc[i, "Variant_Classification"]
                protein_change=gene_ds.loc[i, protein_change_identifier]
                sample_key=sample_dict[gene_ds.loc[i, 'Tumor_Sample_Barcode']]
            
                if "variant" in gene_variant_count[gene]:
                
                    #Add to final_dict
                    if not protein_change is np.nan: 
                        if not gene+'_'+protein_change in final_dict:
                            final_dict[gene+'_'+protein_change] = [0]*len(sample_dict)
                        final_dict[gene+'_'+protein_change][sample_key]=1
                
                if variant_class != 'Silent' and "Nonsilent" in gene_variant_count[gene]:
                    if not gene+'_Nonsilent' in final_dict:
                        final_dict[gene+'_Nonsilent'] = [0]*len(sample_dict)
                    final_dict[gene+'_Nonsilent'][sample_key]=1
        
                if variant_class in gene_variant_count[gene]:
                    if not gene+'_'+variant_class in final_dict:
                        final_dict[gene+"_"+variant_class] = [0]*len(sample_dict)
                    final_dict[gene+'_'+variant_class][sample_key]=1            
                

    pre_table=pd.DataFrame(final_dict, index=sample_dict.keys()).T
    
    if underscore_and_truncate_sample_names:
        print("Editing sample names...")
        column_list=[]
        for i in pre_table.columns:
            column_list.append('_'.join(i.split(sep='-')[:3]))
        pre_table.columns=column_list
        
    print ("Writing to "+ gct_output_file)
#    ccal.write_gct(pre_table, gct_file_path=gct_output_file)
    ccal.write_gct(pre_table, file_path=gct_output_file)
    return pre_table

def make_activation_panel2(
    h_projected,
    h_reference,
    title = None,
    display_changes = False,
    fsize = 10,
    file = None,
    clims = [-1.5, 1.5]):
    
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Rectangle, Arrow
    import numpy as np
    
    C = h_projected
    
    #print(C)
    
    CN1 = (C - C.min(axis=0))/(C.max(axis=0) - C.min(axis=0))
    CN1 = CN1.div(CN1.sum(axis=0))
    D = C.T
    R = h_reference.T
    D = ((D - R.mean(axis=0))/R.std(axis=0)).clip(lower = clims[0], upper = clims[1])
    D  = ((D - clims[0])/(clims[1] - clims[0]))
    CN2 = D.T

    patches = []
    colors = np.array([0.0] * (CN2.shape[0] * CN2.shape[1]))
    cmap = matplotlib.cm.get_cmap('bwr')

    x = list((np.array(range(CN2.shape[1]))/CN2.shape[1]) + 0.055)
    y = list(reversed(list((np.array(range(CN2.shape[0]))/CN2.shape[0]) + 0.055)))

    k = 0
    for i in range(len(x)):
        for j in range(len(y)):
            radius = 0.025*(CN1.iloc[j, i] - CN1.min().min())/(CN1.max().max() - CN1.min().min()) + 0.015
            circle = Circle((x[i],y[j]), radius = radius, facecolor = cmap(CN2.iloc[j, i]), 
                        edgecolor='#000000', lw=2.0)
            colors[k] = CN2.iloc[j, i]
            patches.append(circle)
            k = k + 1
      
    if display_changes:  
        #rint(CN1)
        #rint(CN2)
        k = 0
        for i in range(len(x) - 1):
            for j in range(len(y)):
                l1 = (CN1.iloc[j, i+1]  - CN1.iloc[j, i])/(len(y) * 1.5)
                l2 = (CN2.iloc[j, i+1]  - CN2.iloc[j, i])/(len(y) * 1.5)
                length =  l1 + l2
                #rint('{} {} {}'.format(i, j, length))
                arrow = Arrow(x[i] + 1/(2*CN2.shape[1]), y[j] - length/2, 0, length, width=0.06,
                             color='black')
                patches.append(arrow)
                k = k + 1
               
    #rectangle = Rectangle((0.1, 0.1),0.5,0.5,fill=False)
    #patches.append(rectangle)         
            
    fig = plt.figure(figsize=(fsize, fsize))
    #plt.xlim(-.5, 1.5)
    #plt.ylim(0, 1.5)

    ax = fig.add_subplot(111, aspect='equal', facecolor='w')

    #p = PatchCollection(patches, cmap='bwr', alpha=1)
    p = PatchCollection(patches, match_original=True)
    #p.set_array(colors)
    ax.add_collection(p)

    #plt.colorbar(p)
    ax.set_xlim(-.25, 1.10)
    ax.set_ylim(0, 1.2)
    ax.set_axis_off()

    fontname='Calibri'
    #Helvetica'
    
    if title != None:
        ax.text(0.6, 1.15, title, horizontalalignment='right', verticalalignment='center', fontsize=20,
                fontweight = 'bold', fontname = fontname)
    
    for i in range(len(y)):
        ax.text(-0.075, y[i], C.index[i], horizontalalignment='right', verticalalignment='center', fontsize=14,
                fontweight = 'bold', fontname = fontname)

    for i in range(len(x)):
        ax.text(x[i], 1.0, C.columns[i], horizontalalignment='center', verticalalignment='bottom', fontsize=14,
                fontweight = 'bold', fontname = fontname, rotation = 'vertical')
    

    #cbar = fig.colorbar(ax) #, orientation='horizontal')
    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar    
       
    plt.show()
    
    if file != None:
        fig.savefig(fname = file, dpi=200, bbox_inches='tight')
    #print(CN1)
    #print(CN2)
    
    return
