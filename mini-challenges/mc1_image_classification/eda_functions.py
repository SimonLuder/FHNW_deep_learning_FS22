import plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import wandb


def compare_mlp_sizes(df, variant):
    '''creates a heatmap that shows the best achieved accuracy for a combination of number of layers and number of nodes per layer'''
    df = df.copy()
    if variant:
        df = df.loc[df["variant"]==variant]

    df["hidden_layer_size"] = df["hidden_layers"].apply(lambda x: x[0])
    df = df[["num_hidden", "hidden_layer_size", "test accuracy"]].groupby(["num_hidden", "hidden_layer_size"]).max().reset_index()
    df = df.pivot(index='num_hidden', columns='hidden_layer_size', values='test accuracy')

    fig = go.Figure(data=go.Heatmap(
        z = df.values,
        x = df.columns.astype(str),
        y = df.index.astype(str),
        texttemplate="%{z}",
        textfont={"size":20},
        colorscale = "Greens"))

    fig.update_layout(
        title_text='Best MLP test accuracy scores for varying layers',
        xaxis_title="hidden layer size",
        yaxis_title="number of hidden layers",
        legend_title="test accuracy")
    fig.show()


def compare_cnn_sizes(df, variant):
    '''creates a heatmap that shows the best achieved accuracy for a combination of number of layers and number of nodes per layer'''
    df = df.copy()
    if variant:
        df = df.loc[df["variant"]==variant]

    df["conv_layer_size"] = df["conv_layers"].apply(lambda x: x[0])
    df = df[["num_conv", "conv_layer_size", "test accuracy"]].groupby(["num_conv", "conv_layer_size"]).max().reset_index()
    df = df.pivot(index='num_conv', columns='conv_layer_size', values='test accuracy')

    fig = go.Figure(data=go.Heatmap(
        z = df.values,
        x = df.columns.astype(str),
        y = df.index.astype(str),
        texttemplate="%{z}",
        textfont={"size":20},
        colorscale = "Greens"
        ))

    fig.update_layout(
        title_text='Best CNN test accuracy scores for varying layers',
        xaxis_title="convolutional layer size",
        yaxis_title="number of convolutional layers",
        legend_title="test accuracy",)
    fig.show()


def compare_learning_rates_per_model(df):
    '''Creates a scatterplot to compare model variants, learning rates and the best achieved test accuracy'''
    df = df.loc[(df["variant"]=="MLP_SGD") | (df["variant"]=="CNN_SGD")]
    fig = px.scatter(
        df,
        x = df["variant"].astype(str) + df["conv_layers"].astype(str).replace("nan", "") + df["hidden_layers"].astype(str) ,
        y = df["top test accuracy"],
        color = df["learning_rate"],
        color_continuous_scale="Sunset_r")

    fig.update_layout(
        title_text='Best test accuracy scores for varying layers learning rates per model',
        yaxis_title="best test accuracy",
        xaxis_title="network variants",
        legend_title="learning rate",
        height=800)

    fig.update_traces(marker=dict(size=12))
    fig.show()


def show_training(df, variant, columns):
    ''''Visualize training for a list of metrics
    Args:
        df (pandas DataFrame):
        variant (string): sweep variant
        columns (list): colnames where the metrics are stored
    '''
    if variant:
        df = df.loc[df["variant"]==variant]
    for col in columns:
        show_training_single_plot(df, variant, col)


def show_training_single_plot(df, variant, col):
    '''Visualize training of a single metric'''
    if variant:
        df = df.loc[df["variant"]==variant]

    lrs = sorted(set(df["learning_rate"]))
    color_dict = {key:value for key, value in zip(lrs, plotly.colors.sample_colorscale("Sunset_r", samplepoints=len(lrs), low=0.1, high=1.0,))}
    df["color_lr"] = df["learning_rate"].apply(lambda x: color_dict[x])

    fig = go.Figure()
    # Add traces
    for name in set(df["name"]):
        fig.add_trace(go.Scatter(x=df[df["name"]==name]._step,
                                 y=df[df["name"]==name][col],
                                 mode='lines',
                                 marker_color= list(set(df[df["name"]==name]["color_lr"]))[0],
                                 name=name,
                                 text=df[df["name"]==name]["learning_rate"],
                                 hovertemplate="epoch: %{x}<br>value: %{y}<br>lr: %{text}<br>"))

    fig.update_layout(title_text="{} over epochs".format(col),
                      xaxis_title="epochs",
                      yaxis_title=col,
                      height=400)
    fig.update_yaxes(range=[np.min(df[col].astype(float)), np.min([3.5, np.max(df[col].astype(float))])])

    fig.show()

def plot_train_vs_test_loss(df, variant):
    if "DP" in variant:
        col = "drop_out"
    if "L2" in variant:
        col = "l2_weight_decay"

    df = df.loc[df["variant"]==variant]
    fig = go.Figure()
    # Add traces
    for name in sorted(set(df[col])):
        fig.add_trace(go.Scatter(x=df[df[col]==name]["train loss"],
                                 y=df[df[col]==name]["test loss"],
                                 mode='markers+lines',
                                 name=set(df[df[col]==name][col]).pop()))

    fig.update_layout(title_text="Development of test loss in regard to training loss for {}".format(variant),
                      xaxis_title="train loss",
                      yaxis_title="test loss",
                      legend_title=col,

                      xaxis_autorange="reversed",
                      height=450)
    fig.show()


def plot_train_vs_test_accuracy(df, variant):
    if "DP" in variant:
        col = "drop_out"
    if "L2" in variant:
        col = "l2_weight_decay"

    df = df.loc[df["variant"]==variant]
    fig = go.Figure()
    # Add traces
    for name in sorted(set(df[col])):
        fig.add_trace(go.Scatter(x=df[df[col]==name]["train accuracy"],
                                 y=df[df[col]==name]["top test accuracy"],
                                 mode='markers+lines',
                                 name=set(df[df[col]==name][col]).pop()))

    fig.update_layout(title_text="Development of top test accuracy in regard to training accuracy for {}".format(variant),
                      xaxis_title="train accuracy",
                      yaxis_title="top test accuracy",
                      legend_title=col,
                      height=450)
    fig.show()


def max_test_acc_reg_bar(df, variant):
    if "DP" in variant:
        reg_title = "drop out"
        col = "drop_out"
    if "L2" in variant:
        reg_title = "l2"
        col = "l2_weight_decay"

    df = df.copy()
    df = df.loc[df["variant"]==variant]
    df = df.groupby("name").max("_step")
    df = df.sort_values(col)
    fig = go.Figure([go.Bar(x=df[col].astype(str), y=df["top test accuracy"], text=df["top test accuracy"])])

    fig.update_layout(title_text="Top accuracy for varying {} values".format(reg_title),
                      xaxis_title="{} regularization strength".format(reg_title),
                      yaxis_title="top test accuracy",
                      barmode='group',
                      height=400)
    fig.show()


def read_wandb_logs():
    '''reads wandb logs of all runs'''
    api = wandb.Api()
    entity, project = "simonluder", "del-mc1"  # set to your entity and project
    runs = api.runs(entity + "/" + project)

    summary_list, config_list, name_list, history_list = [], [], [], []
    for i, run in enumerate(runs):
        # if i % 20 == 0:
        #     print("Reading model:", i)
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # .history creates a history dataframe of all logged metrics
        history_list.append(run.history())
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})
        # .name is the human-readable name of the run.
        name_list.append(run.name)

    # combine into single dataframes
    df_history_summary = pd.DataFrame()
    df_config_summary = pd.DataFrame()
    for i, name in enumerate(name_list):
        history_list[i]["name"] = name
        df_history_summary = pd.concat([df_history_summary, history_list[i]])
        config_list[i]["name"] = name
        df_config_summary = df_config_summary.append(config_list[i], ignore_index=True)
    df_history_summary = df_history_summary.reset_index(drop=True)
    df_config_summary.loc[df_config_summary["variant"] == "MLP_SGD_Adam", "variant"] = "MLP_Adam"
    df_config_summary.loc[df_config_summary["variant"] == "CNN_SGD_Adam", "variant"] = "CNN_Adam"
    return df_config_summary, df_history_summary


def max_test_acc_lr_bar(df, variant):
    col = "learning_rate"
    df = df.copy()
    df = df.loc[df["variant"]==variant]
    df = df.groupby("name").max("_step")
    df = df.sort_values(col)
    fig = go.Figure([go.Bar(x=df[col].astype(str), y=df["top test accuracy"], text=df["top test accuracy"])])

    fig.update_layout(title_text="Top accuracy for varying learning rates for {}".format(variant),
                      xaxis_title="learning rate",
                      yaxis_title="top test accuracy",
                      barmode='group',
                      height=400)
    fig.show()

def print_best_models(df, variant=None):
    '''Prints the best model vor a specified variant'''
    # select model variant
    if variant:
        df = df.loc[df["variant"]==variant]
    # select model with best test accuracy
    df = df.loc[df["top test accuracy"]==np.max(df["top test accuracy"])]
    print("Best \033[1m{}\033[0m model reached a test accuracy of \033[1m{}\033[0m at epoch \033[1m{}\033[0m with and test loss of \033[1m{:.3f}\033[0m".format(
        df["variant"].values[0],
        df["test accuracy"].values[0],
        df["_step"].values[0].astype(int),
        df["test loss"].values[0]))
