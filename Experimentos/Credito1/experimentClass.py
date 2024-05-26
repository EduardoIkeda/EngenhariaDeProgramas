from time import time
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output, display, Markdown, Latex

class Experiment:
    def __init__(self, repetitions, data_quantity):
        self.repetitions = repetitions
        self.data_quantity = data_quantity
        self.results = []         #T
        self.variability = []     #V
        self.mean = []            #M
        self.std_deviation = []   #S
        
    
    def run_experiment(self, experiment_function):
        n = 1
        for i in range(self.repetitions):
            tic = time()
            # Run the experiment function
            for j in range(self.data_quantity):
                experiment_function()
            tac = time()
            self.results.append(tac - tic)
            self.std_deviation.append(np.std(self.results))
            self.mean.append(np.mean(self.results))
            self.variability.append(np.std(self.results) / np.mean(self.results))
            clear_output(wait=True)
            print("Processing data... {:.2%}".format((i + 1) / self.repetitions))
        display(Markdown("<center><h2><font color='green'>Terminou!</font></h2></center>"))

    @staticmethod
    def plot_title(title="", additional_title=""):
        if additional_title:
            title += f' - {additional_title}'
        plt.title(title)
        
    @staticmethod
    def show_bar_plot(experiment, additional_title="", color_variable='darkblue'):
        plt.bar(range(experiment.repetitions), experiment.results, color=color_variable)
        plt.xlabel('Experiment')
        plt.ylabel('Time (s)')
        Experiment.plot_title("Experiment result", additional_title)
        plt.show()

    @staticmethod
    def setup_histogram(experiment, color_variable='darkblue'):
        plt.hist(experiment.results, bins=20, color=color_variable)
        plt.xlabel('Time')
        plt.ylabel('Frequency')

    @staticmethod
    def show_histogram(experiment, additional_title="", color_variable='darkblue'):
        plt.figure(figsize=(5,3))
        Experiment.setup_histogram(experiment, color_variable)
        Experiment.plot_title("Histogram", additional_title)
        plt.grid()
        plt.show()

    @staticmethod
    def show_histograms_side_by_side(experiment1, experiment2, additional_title="", legend="", color_variable1='darkblue', color_variable2='orange'):
        plt.figure(figsize=(5,3))
        Experiment.setup_histogram(experiment1, color_variable1)
        Experiment.setup_histogram(experiment2, color_variable2)
        plt.legend(legend)
        Experiment.plot_title(additional_title)
        plt.grid()
        plt.show()
        
    @staticmethod
    def show_variability_plot(experiment, additional_title="", color_variable1='darkblue'):
        plt.figure(figsize=(5,3))
        plt.plot(experiment.variability, color=color_variable1)
        plt.xlabel('Experiments')
        plt.ylabel('Average Time')
        Experiment.plot_title("Variability", additional_title)
        plt.grid()
        plt.show()

    @staticmethod
    def show_variability(experiment, title=""):
        std_deviation = np.std(experiment.results)
        mean = np.mean(experiment.results)
        variability = std_deviation/mean
        display(Markdown("<center><h2>" + title +"</h2></center>"))
        display(Markdown("<center><h3>" + "{:}".format(variability) +"</h3></center>"))

    @staticmethod
    def show_mean_deviation_plot(experiments, additional_title="", step=0.05):
        plt.plot(experiments.mean)
        plt.plot(experiments.std_deviation)
        plt.yticks(np.arange(0, max(experiments.results) + step, step))
        Experiment.plot_title("Mean vs Deviation", additional_title)
        plt.grid()
        plt.legend(["Mean", "Standard Deviation"])
        plt.show()
        
    @staticmethod
    def setup_scatter(experiment, color_variable):
        experiment_mean = np.mean(experiment.results)
        plt.plot(range(experiment.repetitions), experiment.results, color_variable + '.')
        plt.xlabel('Experiment Repetition')
        plt.ylabel('Time of '+str(experiment.data_quantity)+' expl. attrib.')

    @staticmethod
    def show_scatter_plot(experiment, color_variable, additional_title=""):
        plt.figure(figsize=(5,3))
        Experiment.setup_scatter(experiment, color_variable)
        Experiment.plot_title("Scatter plot", additional_title)
        plt.grid()
        plt.show()

    @staticmethod
    def show_scatter_plot_side_by_side(experiment1, experiment2, color_variable1, color_variable2, legend1, legend2, additional_title=""):
        plt.figure(figsize=(5,3))
        
        Experiment.setup_scatter(experiment1, 'b')
        Experiment.setup_scatter(experiment2, 'r')

        Experiment.plot_title("Scatter plot", additional_title)
        
        plt.legend(["atrib. expl."+ legend1 + " Tmedio ={:.3g}".format(np.mean(experiment1.results))+"seg",
                   "atrib. expl." + legend2 + " Tmedio ={:.3g}".format(np.mean(experiment2.results))+"seg"])
        
        plt.xlabel('Experiment Repetition')
        plt.ylabel('Time of '+str(experiment1.data_quantity)+' expl. attrib.')
        plt.grid()
        plt.show()

    