# Helper functions for classifier model analysis
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

def get_predictions(path, classes=None):
    ''' Returns predictions dataframe with class probabilities '''
    df = pd.read_csv(path)
    if classes is not None:
        cols1 = ['file.path', 'patient.number', 'laterality', 'True Class', 'Predicted Class']
        cols2 = classes
        df = pd.concat([df[cols1], df[cols2]], axis=1)
    return df

def make_confusion_matrix(df, visualize=True, save=None, **kwargs):
    predictions = df['Predicted Class'].values
    targets = df['True Class'].values
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, predictions, normalize='true', labels=kwargs['labels'] if 'labels' in kwargs.keys() else None)
    
    if visualize:
        labels = kwargs['labels'] if 'labels' in kwargs.keys() else None
        title = kwargs['title'] if 'title' in kwargs.keys() else None
        plot_cm(cm, labels, title, save=save)
        
    return cm

class AUROC():
    def __init__(self, df, labels):
        self.df = df.copy(deep=True)
        self.nclasses = len(df['True Class'].unique())
        self.labels = labels
        assert len(self.labels) == self.nclasses
        
    def binarize(self, c):
        df = self.df.copy(deep=True)
        df['True Class'] = list(map(int, list(df['True Class'] == c)))
        return df
        
    def make_boot(self, df):
        df = df.copy(deep=True)
        idxs = np.random.randint(low=0, high=len(df), size=(len(df),))
        while len(df.iloc[idxs]['True Class'].unique()) < 2:
            idxs = np.random.randint(low=0, high=len(df), size=(len(df),))
        df = df.iloc[idxs]
        return df

    def __call__(self, nboot=500, random_state=None):
        from sklearn.metrics import roc_auc_score

        # bootstrapped samples
        sample_aucs = np.zeros((self.nclasses, nboot))
        
        # stratify by class
        for i, c in enumerate(self.labels):
            
            if random_state is not None:
                np.random.seed(random_state)
            
            # binarize as class vs no class
            bin_df = self.binarize(c)
        
            # compute auroc bootstrapped
            for j in range(nboot):
                
                # get a single bootstrapped sample
                sample = self.make_boot(bin_df)

                # compute auroc on sample
                auc = roc_auc_score(sample['True Class'], sample[f'Prob_{c}'])
                
                # append to matrix
                sample_aucs[i, j] = auc
                
        # estimate 95% CI
        sample_aucs = np.sort(sample_aucs, axis=1)
        lower, upper = int(0.025*nboot), int(0.975*nboot)
        per_class_statistics = np.concatenate([np.mean(sample_aucs, axis=1)[:, None], sample_aucs[:, lower][:, None], sample_aucs[:, upper][:, None]], axis=1)
        per_class_statistics = pd.DataFrame(per_class_statistics, columns=['auroc', 'lower', 'upper'])
        per_class_statistics['Class'] = self.labels
        
        # estimate macro auc with 95% CI
        macro_avg = np.mean(sample_aucs, axis=0)
        macro_avg = np.sort(macro_avg)
        assert len(macro_avg) == nboot
        macro_statistics = pd.DataFrame(data={'auroc': [np.mean(macro_avg)], 'lower': [macro_avg[lower]], 'upper': [macro_avg[upper]]})
        
        return per_class_statistics, macro_statistics
                
# visualization functions
def plot_cm(cm, labels=None, title=None, save=None):
    ''' Plot the confusion matrix as a heatmap '''
    from sklearn.metrics import ConfusionMatrixDisplay
    
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(20,20))
    disp.plot(ax=ax, values_format='.2f', colorbar=False)
    plt.xticks(rotation=45)
    plt.colorbar(disp.im_, fraction=0.046, pad=0.04)
#     sns.heatmap(cm, fmt='d', annot=True, square=True,
#                 xticklabels=labels, yticklabels=labels,
#                 cmap='gray_r', vmin=0, vmax=0,  # set all to white
#                 linewidths=0.5, linecolor='k',  # draw black grid lines
#                 cbar=False)                     # disable colorbar

#     # re-enable outer spines
#     sns.despine(left=False, right=False, top=False, bottom=False)
    plt.title(title)
    if save is not None:
        plt.savefig(save)
    plt.show()
    plt.close()
    
def make_barplot(cm, labels=None, title=None, normalize=True, order=None, save=None):
    ''' Plot barplots representing per-class prediction frequencies '''
    pred_freq = np.sum(cm, axis=0)
    true_freq = np.sum(cm, axis=1)
    
    if normalize:
        pred_freq = pred_freq / np.sum(pred_freq)
        true_freq = true_freq / np.sum(true_freq)
    
    pred_freq = pd.DataFrame(pred_freq, columns=['Frequency'])
    true_freq = pd.DataFrame(true_freq, columns=['Frequency'])
    pred_freq['Class'] = labels
    true_freq['Class'] = labels
    true_freq['Type'] = ['True']*len(true_freq)
    pred_freq['Type'] = ['Pred']*len(pred_freq)
    data = pd.concat([pred_freq, true_freq], axis=0)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='Class', y='Frequency', hue='Type', order=order)
    plt.xticks(rotation=45)
    plt.title(title)
    if save is not None:
        plt.savefig(save)
    plt.show()
    plt.close()

def plot_roc(nrows, ncols, models, model_stats, labels, title):
    
    from sklearn.metrics import roc_curve, roc_auc_score
    
    assert nrows*ncols >= len(labels)

    # create figure
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(3*ncols, 3*nrows))
    
    for i, cls in enumerate(labels):
        
        r = i // ncols
        c = i % ncols

        # plot model ROC curves
        for m in models.keys():
            df = models[m].copy(deep=True)
            df['True Class'] = list(map(int, list(df['True Class'] == cls)))
            fpr, tpr, _ = roc_curve(df['True Class'], df[f'Prob_{cls}'])
            stats = model_stats[m][0]
            axs[r,c].plot(fpr, tpr, alpha=1., lw=1, label='{}: {:.2f}'.format(m.split('_')[0], stats[stats['Class'] == cls]['auroc'].item()))
            
        # define auc in legend
        axs[r,c].legend(title='Area under Curve')
            
        # plot low skill classifier
        axs[r,c].title.set_text(cls)
        axs[r,c].plot([0, 1], [0, 1],'r:')
        axs[r,c].set_xlim([0, 1])
        axs[r,c].set_ylim([0, 1.05])
        axs[r,c].set_ylabel('True Positive Rate')
        axs[r,c].set_xlabel('False Positive Rate')
        
    fig.subplots_adjust(wspace=0.05)
    fig.suptitle(title, fontsize=16, y=1)
    
    plt.tight_layout()
    plt.show()
    plt.close()