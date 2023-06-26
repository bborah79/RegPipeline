import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class VisualizationData:
    '''Creates various vizualization plots.

    Parameters
    ----------
    results : a pandas dataframe
        A dataframe of all the accumulated performance metrics of all the
        models tried.

    Methods
    -------
    plot_train_test_scores():
        Plots the train and test scores for the whole data set.

    plot_cv_train_test_scores():
        Plots the cross-validated train and test scores of the models.

    plot_pvalues():
        Plots the p-values of the ttest performed on the cross-validated
        scores.

    '''

    def __init__(self, results):
        self.results = results

    def plot_train_test_scores(self):
        '''Plots train test scores on the whole dataset.

        '''

        results1_score = self.results.drop(
            self.results.filter(regex="r2|%diff|cv|pval|t-stat").columns, axis=1
        )
        results1_score.plot(kind="bar", figsize=(12, 6))
        plt.title(
            "Train-test scores of the models trained on the whole training set and tested on the holdout set"
        )
        plt.xlabel("Models")

        plt.savefig("./output/train_test_scores_fulldata.png", bbox_inches="tight")

    def plot_cv_train_test_scores(self):
        '''Plots cross-validated train test scores.

        '''

        results0_score = abs(self.results.filter(regex="^cv.*r2$", axis=1))
        results1_score = abs(self.results.filter(regex="^cv.*mse$", axis=1))
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        results0_score.plot(kind="bar", ax=axes[0], xlabel="Models")
        axes[0].set_title("Cross validated mean train-test r2 scores")
        axes[0].set_xlabel("Models", labelpad=6.0, fontsize=16.0, fontweight="bold")
        axes[0].set_ylabel("r2 values", fontsize=16.0, fontweight="bold")
        results1_score.plot(kind="bar", ax=axes[1], xlabel="Models")
        axes[1].set_title("Cross validated mean train-test MSE")
        axes[1].set_xlabel("Models", labelpad=6.0, fontsize=16.0, fontweight="bold")
        axes[1].set_ylabel("MSE", fontsize=16.0, fontweight="bold")

        fig.savefig("./output/cv_train_test_scores.png", bbox_inches="tight")

    def plot_pvalues(self):
        '''Plots p-values of ttest performed on the cross-validated scores

        '''

        results0_pval = self.results.filter(regex="pval", axis=1)
        results0_pval.plot(kind="bar", figsize=(12, 6))
        plt.title(
            "p-values of t-test comparing the mean of the cross validated train-test scores"
        )
        plt.xlabel("Models")
        plt.savefig("./output/cv_pval.png", bbox_inches="tight")

    def plot_confidence_intervals(self):
        xci_train_plot = np.zeros((2, int(self.results[0].shape[0])))
        xci_test_plot = np.zeros((2, int(self.results[0].shape[0])))
        xci_train_plot[0] = abs(np.array([val[0] for val in self.results[2]]))
        xci_train_plot[1] = np.array([val[2] for val in self.results[2]])
        xci_test_plot[0] = abs(np.array([val[0] for val in self.results[3]]))
        xci_test_plot[1] = np.array([val[2] for val in self.results[3]])
        mean_pred_train_plot = np.array([val[1] for val in self.results[2]])
        mean_pred_test_plot = np.array([val[1] for val in self.results[3]])
        xci = np.arange(len(mean_pred_train_plot))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(
            mean_pred_train_plot,
            xci,
            xerr=xci_train_plot,
            fmt="o",
            capsize=10.0,
            label="Training CI",
        )
        ax.errorbar(
            mean_pred_test_plot,
            xci,
            xerr=xci_test_plot,
            fmt="o",
            capsize=10.0,
            label="Testing CI",
        )
        ax.set_xlabel("Mean Predicted Value")
        ax.set_ylabel("Model id")
        ax.legend()
        ax.set_title("Confidence interval of the predicted mean of each model")
        plt.show
