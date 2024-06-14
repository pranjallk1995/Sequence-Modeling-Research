""" module to plot training history """

import config as cfg
import logging as lg
import tensorflow as tf
import plotly.graph_objs as go

class Plotting():
    """ class to plot the loss over epochs """

    def __init__(self, loss_history: tf.keras.callbacks.History , save_location: str) -> None:
        self.history = loss_history
        self.save_file = save_location

    def plot_loss(self) -> go.Figure:
        """ function to return the loss trace """
        return [
            go.Scatter(
                x=self.history.history["loss"], y=[epoch for epoch in range(cfg.TOTAL_EPOCHS)],
                mode="markers", name="loss"
            )
        ]

    def plot(self) -> None:
        """ module entry point """
        lg.info("Model loss plot saved in %s", self.save_file)
        loss_trace = self.plot_loss()
        fig = go.Figure(loss_trace)
        fig.update_layout(
            title="Model Loss",
            xaxis_title="X axis",
            yaxis_title="Y axis",
            showlegend=False,
            template="plotly_dark",
        )
        fig.write_html(self.save_file)
