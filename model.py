from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras

class Predictor:

    def __init__(self):
        self.train_data = None
        self.train_features = None
        self.train_labels = None
        self.valid_data = None
        self.valid_features = None
        self.valid_labels = None
        self.model = None
        self.history = None
        self.col_names = ['WL_400', 'WL_412', 'WL_442', 'WL_490', 'WL_510', 'WL_560',
                          'WL_620', 'WL_665', 'WL_673', 'WL_681', 'WL_708', 'WL_753',
                          'WL_761', 'WL_764', 'WL_767', 'WL_778', 'CHL']

    def load_data(self, train_name, valid_name):
        self.train_data = pd.read_csv(train_name, header=0, names=self.col_names)
        self.valid_data = pd.read_csv(valid_name, header=0, names=self.col_names)

    def process_data(self, filter, threshold):
        self.train_features = self.train_data.copy()
        self.train_features = self.train_features[self.train_features[filter] > threshold]
        self.train_features.pop(filter)
        self.train_labels = self.train_features.pop(self.col_names[-1])
        self.valid_features = self.valid_data.copy()
        self.valid_features = self.valid_features[self.valid_features[filter] > threshold]
        self.valid_features.pop(filter)
        self.valid_labels = self.valid_features.pop(self.col_names[-1])
        sf = 0.9 / self.train_features.max().max()
        self.train_features *= sf
        self.valid_features *= sf

    def setup_model(self, units, regularization, dropout, verbose=False):
        self.model = keras.Sequential([
            keras.Input(shape=(self.train_features.shape[1],)),
            keras.layers.Dense(
                units=units,
                activation=keras.activations.linear,
                kernel_regularizer=keras.regularizers.l2(regularization)
            ),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(
                units=units,
                activation=keras.activations.linear,
                kernel_regularizer=keras.regularizers.l2(regularization)
            ),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1)
        ])
        if verbose:
            self.model.summary()

    def compile_model(self, learning_rate):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError()
        )

    def train_model(self, batch_size, epochs):
        self.history = self.model.fit(
            self.train_features, self.train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(self.valid_features, self.valid_labels)
        )

    def save_model(self, model_name):
        self.model.save(model_name)
        history = pd.DataFrame(self.history.history)
        history.to_csv(model_name + '/history.csv', mode='w')

    def load_model(self, model_name):
        self.model = keras.models.load_model(model_name)

    def predict_test(self, test_name, filter):
        test_data = pd.read_csv(test_name, header=0, names=self.col_names[:-1])
        test_data.pop(filter)
        results = pd.DataFrame(self.model.predict(test_data), columns=self.col_names[-1:])
        results.to_csv('test_results.csv', mode='w')

    def predict_valid(self, save=False):
        x = self.valid_labels
        y = self.model.predict(self.valid_features).flatten()
        plt.figure(figsize=(4, 4))
        plt.plot(x, y, 'go')
        lim = max(x.max(), y.max())
        plt.xlim(0, lim)
        plt.ylim(0, lim)
        plt.xticks(np.arange(0, lim + 1, 10))
        plt.yticks(np.arange(0, lim + 1, 10))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplots_adjust(left=0.13, right=0.93, top=0.93, bottom=0.13)
        plt.title('Validation')
        plt.xlabel('True CHL')
        plt.ylabel('Predicted CHL')
        plt.grid(True)
        if save:
            plt.savefig('plot_prediction')
            plt.close()
        else:
            plt.show()

    def plot_history(self, save=False):
        plt.plot(self.history.history['loss'], label='training')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend(frameon=False)
        plt.grid(True)
        if save:
            plt.savefig('plot_history')
            plt.close()
        else:
            plt.show()

    def plot_chl(self, save=False):
        hist_data = self.train_labels
        plt.hist(hist_data, bins=50, range=(0.0, 100.0), color='b', alpha=0.7)
        plt.subplots_adjust(left=0.13, right=0.93, top=0.93, bottom=0.13)
        plt.title(self.col_names[-1])
        plt.xlabel('Level')
        plt.ylabel('Entries')
        if save:
            plt.savefig('hist_chl')
            plt.close()
        else:
            plt.show()

    def plot_feature(self, col_name, save=False):
        hist_data = self.train_features[col_name]
        plt.hist(hist_data, bins=50, range=(0.0, 1.0), color='g', alpha=0.7)
        plt.subplots_adjust(left=0.13, right=0.93, top=0.93, bottom=0.13)
        plt.title(col_name)
        plt.xlabel('Rrs')
        plt.ylabel('Entries')
        hist_txt = '$\mu = {:.3f}$'.format(hist_data.mean())
        plt.text(0.9, 0.90, hist_txt, ha='right', va='top', transform=plt.gca().transAxes)
        hist_txt = '$\sigma = {:.3f}$'.format(hist_data.std())
        plt.text(0.9, 0.84, hist_txt, ha='right', va='top', transform=plt.gca().transAxes)
        hist_txt = 'median = {:.3f}'.format(hist_data.median())
        plt.text(0.9, 0.78, hist_txt, ha='right', va='top', transform=plt.gca().transAxes)
        if save:
            plt.savefig('hist_' + col_name)
            plt.close()
        else:
            plt.show()

    def plot_chl_vs_feature(self, col_name, save=False):
        x = self.train_features[col_name]
        y = self.train_labels
        plt.hist2d(x, y, bins=50, range=[[0.0, 1.0], [0.0, 100.0]],
                   cmap='rainbow', norm=LogNorm(), alpha=0.7)
        plt.subplots_adjust(left=0.1, right=0.89, top=0.9, bottom=0.1)
        plt.title('CHL vs ' + col_name)
        plt.xlabel(col_name)
        plt.ylabel('CHL')
        plt.colorbar(cax=plt.axes([0.9, 0.1, 0.03, 0.8]))
        if save:
            plt.savefig('hist_chl_vs_' + col_name)
            plt.close()
        else:
            plt.show()

def main():

    run_test = False
    plot_data = False
    save_plots = False
    pred = Predictor()

    if run_test:
        pred.load_model(model_name='model_nn16_linear_lr0001_bs20_rr001_dp01_ep40')
        pred.predict_test(test_name='testing.csv', filter='WL_560')
        return

    pred.load_data(train_name='training.csv', valid_name='validation.csv')
    pred.process_data(filter='WL_560', threshold=0.018)

    pred.setup_model(units=16, regularization=0.001, dropout=0.1, verbose=True)
    pred.compile_model(learning_rate=0.0001)
    pred.train_model(batch_size=20, epochs=30)
    pred.save_model(model_name='model')

    pred.plot_history(save_plots)
    pred.predict_valid(save_plots)

    if plot_data:
        pred.plot_chl(save_plots)
        for name in pred.col_names[:-1]:
            pred.plot_feature(name, save_plots)
            pred.plot_chl_vs_feature(name, save_plots)

if __name__ == '__main__':
    main()
