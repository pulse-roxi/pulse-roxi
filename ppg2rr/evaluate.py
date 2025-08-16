"""Functions to evaluate algorithm results with accompanying figures."""
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from math import ceil
import warnings
from datetime import datetime


def bland_altman(
    df: pd.DataFrame,
    reference, 
    observed, 
    reference_label = "Reference", 
    observed_label = "Observed", 
    x_range: Optional[list] = None,
    y_range: Optional[list] = None,
    width: Optional[int] = 600,
    height: Optional[int] = 500,
    save_as: Optional[str] = None,
):
    """Generate a Bland-Altman plot using Plotly.

    On the y-axis, bland-altman analysis shows the difference between observed and
    reference values, from which we calculate the
    * bias (mean) +/- 95% confidence interval
    * 95% limit of agreement, +/- 95% confidence interval

    On the x-axis is shown the average of observed and reference. (Bland and Altman
    demonstrated in https://doi.org/10.1016/S0140-6736(95)91748-9 why you should not 
    use the reference as x, as is sometimes done.)

    Assuming n > 30, and we can use the Z-score to calculate confidence intervals.
    For n < 30, the t-score is needed to calculate confidence intervals.
    For additional reference, see
    https://rowannicholls.github.io/python/statistics/confidence_intervals.html

    Parameters:
        df (DataFrame): used to supply the hover text
        reference (list or numpy array): Reference data.
        observed (list or numpy array): Observed Data.
        reference_label: text to describe the reference.
        observed_label: text to describe the observed.
        x_range: list of min and max. Default: auto range.
        y_range: list of min and max. Default: auto range.
        save_as: If set, the plot is saved as that path and name. A timestamp and .png 
            are appended. Default: None (not saved).

    Returns:
        A Plotly figure object.
    """
    diff = observed - reference
    mean = (reference + observed)/2

    # Add horizontal lines for the mean difference and limits of agreement
    bias = np.nanmean(diff)
    std_diff = np.nanstd(diff, axis=0)
    upper_loa = bias + 1.96 * std_diff
    lower_loa = bias - 1.96 * std_diff

    # Confidence interval calculation based on
    # https://rowannicholls.github.io/python/statistics/agreement/bland_altman.html

    n = reference.shape[0]  # Sample size
    se_bias = bias / np.sqrt(n)  # Standard error of the bias
    se_loas = np.sqrt(3 * bias**2 / n)  # Standard error of the LOAs

    # Confidence interval for the bias
    ci_bias = bias - 1.96 * se_bias, bias + 1.96 * se_bias
    # Confidence interval for the lower LOA
    ci_lower_loa = lower_loa - 1.96 * se_loas, lower_loa + 1.96 * se_loas
    # Confidence interval for the upper LOA
    ci_upper_loa = upper_loa - 1.96 * se_loas, upper_loa + 1.96 * se_loas

    # Hover text
    try:
        hover_cols = df[["trial", "dataset-id", "frame index", reference_label, observed_label]]
        hovertemplate_txt = """session: %{text[0]} %{text[1]}
                    <br>frame: %{text[2]}
                    <br>reference: %{text[3]:.2f}
                    <br>observed:  %{text[4]:.2f}"""
    except:
        hover_cols = None
        hovertemplate_txt = None

    fig = go.Figure()

    # Simulate the axes of the underlying scatter plot
    # TODO: Use plot's limits
    fig.add_trace(go.Scatter(
        x=[20,0,40],
        y=[40,0,-80],
        mode="lines",
        line=dict(
            color='white',
            width=1.5,
        ),
    ))

    # Plot
    fig.add_trace(go.Scatter(
        x=mean, 
        y=diff, 
        text=hover_cols,
        hovertemplate=hovertemplate_txt,
        mode="markers",
        marker=dict(
            symbol='circle',
            color='navy',
            opacity=0.2,
            size=4
        )
    ))

    # Linear regression
    mask = ~np.isnan(mean) & ~np.isnan(diff)
    x_data = mean[mask]
    y_data = diff[mask]
    x_data_reshape = np.array(x_data).reshape((-1, 1))

    model = LinearRegression().fit(x_data_reshape, y_data)
    # y_fit = model.predict(x_data_reshape)
    # r_sq  = model.score(x_data_reshape, y_data)

    x_plot = [np.min(x_data), np.max(x_data)]
    y_plot = model.predict(np.array(x_plot).reshape((-1, 1)))

    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        # name=f'linear, r² = {r_sq:.2}',
        name=f'linear regression',
        mode='lines',
        marker=dict(color='orange'),
    ))

    fig.add_hline(
        y=bias,
        line={"color": "black", "width": 1, "dash": "solid"},
        # annotation_text=f"&mu; = {round(bias,2)} ({round(ci_bias[0],2)}, {round(ci_bias[1],2)})",
        # annotation_position="top right",
        annotation_text=f"{bias:6.2f} bias",
        annotation_position="right"
    )
    fig.add_hline(
        y=upper_loa,
        line={"color": "black", "width": 1, "dash": "dash"},
        # annotation_text=f"&mu;+1.96σ = {round(upper_loa, 2)} ({round(ci_upper_loa[0],2)}, {round(ci_upper_loa[1],2)})",  # noqa E501
        # annotation_position="top right",
        annotation_text=f"{upper_loa:6.2f} UL",
        annotation_position="right"

    )
    fig.add_hline(
        y=lower_loa,
        line={"color": "black", "width": 1, "dash": "dash"},
        # annotation_text=f"&mu;-1.96σ = {round(lower_loa, 2)} ({round(ci_lower_loa[0],2)}, {round(ci_lower_loa[1],2)})",  # noqa E501
        # annotation_position="top right",
        annotation_text=f"{lower_loa:6.2f} LL",
        annotation_position="right"
    )

    # Add axis labels and title
    title_text = f"Bland-Altman accuracy"
    sub_title = (
        # r"bias (&mu;) and 95% limits of agreement, with (95% confidence intervals) and linear regression"
        # r"bias, 95% limits of agreement (UL, LL), and linear regression"
        f"{observed_label} vs. {reference_label}"
    )

    fig.update_layout(
        xaxis_title=f"(Observed + Reference) / 2 (bpm)",
        yaxis_title=f"Observed - Reference (bpm)",
        title=f"{title_text}<br><span style='font-size:12px'>{sub_title}</span>",
        xaxis = dict(
            # dtick = 5,
            # constrain = 'domain',
            # constraintoward = 'left'
            # range = [10, 120]
        ),
        yaxis = dict(
            # dtick = 5,
            # scaleanchor = 'x',      # Configure axes to have equal scale
            # scaleratio = 1,
            # range = [-55, 35]
        ),
        showlegend = False,
        width  = width,
        height = height
    )
    if x_range:
        fig.update_layout(xaxis = dict(range = x_range))
    if y_range:
        fig.update_layout(yaxis = dict(range = y_range))

    # Save
    if save_as:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        fig.write_image(f"{save_as}_{current_time}.png", scale=2)

    return fig, (bias, upper_loa, lower_loa)


def violin_comparison(
    y1: np.ndarray[float],
    y2: np.ndarray[float],
    y1_label: str,
    y2_label: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    hover_text_columns: pd.DataFrame = None,
    show_slope_threshold: Optional[float] = None,
):
    """Creates a combined box-and-violin plot for two sets of data points.

    Inspired by https://github.com/jorvlan/open-visualizations, this visualization
    shows how individual data points differ between two sets of data, which helps with
    comparing the behavior of data across two different conditions.

    The size of y1 and y2 must be equal.

    Args:
        y1: data array.
        y2: data array.
        y1_label: str,
        y2_label: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        hover_text_columns: pd.DataFrame = None,
        show_slope_threshold (float): Only lines with slopes greater than the
            show_slope_threshold will be shown. If None (default), show_slope_threshold
            is set to 25% of the std of y1.

    """
    assert len(y1) == len(y2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    n_points = len(y1)
    x1 = np.ones(n_points)
    x2 = np.ones(n_points) * 1.5

    fig = go.Figure()

    # Add line segments connecting the points in x1 and x2
    if show_slope_threshold is None:
        show_slope_threshold = 0.25 * np.std(y1[~np.isnan(y1)])  # ignore nan

    for i in range(len(x1)):
        # only add line if difference is large, otherwise the plot gets too messy
        # when there are too many datapoints
        jitter = np.random.rand() * 0.15
        if abs(y2[i] - y1[i]) > show_slope_threshold:
            if abs(y2[i]) < abs(y1[i]):
                line_color = "rgba(255, 140, 0, 0.603)"
            else:
                line_color = "rgba(30, 144, 255, 0.603)"
            fig.add_trace(
                go.Scatter(
                    x=[x1[i] + jitter, x2[i] - jitter],
                    y=[y1[i], y2[i]],
                    mode="lines+markers",
                    line={
                        "width": 1,
                        "color": line_color,
                    },
                    showlegend=False,
                )
            )
        else:
            # make line very subtle
            fig.add_trace(
                go.Scatter(
                    x=[x1[i] + jitter, x2[i] - jitter],
                    y=[y1[i], y2[i]],
                    mode="lines+markers",
                    line={
                        "width": 0.2,
                        "color": "rgba(192, 210, 241, 0.4)",
                    },
                    showlegend=False,
                )
            )

    hovertemplate_txt = ""
    if hover_text_columns is not None:
        hover_cols = hover_text_columns
    else:
        hover_cols = pd.DataFrame(
            {
                y1_label: np.round(y1, 2),
                y2_label: np.round(y2, 2),
            }
        )

    for idx, col in enumerate(hover_cols):
        hovertemplate_txt += f"{col}:%{{text[{idx}]}}<br>"

    # Create violin plots for each condition
    fig.add_trace(
        go.Violin(
            x=x1,
            y=y1,
            side="negative",
            line_color="dodgerblue",
            opacity=0.9,
            box_visible=True,
            width=0.4,
            meanline_visible=True,
            spanmode="soft",
            points=False,
            # pointpos=0.5,
            name=y1_label,
            # showlegend=True,
            text=hover_cols,
            hovertemplate=hovertemplate_txt,
        )
    )

    fig.add_trace(
        go.Violin(
            x=x2,
            y=y2,
            side="positive",
            line_color="darkorange",
            opacity=0.9,
            box_visible=True,
            width=0.4,
            meanline_visible=True,
            spanmode="soft",
            points=False,
            # pointpos=-0.5,
            name=y2_label,
            # showlegend=True,
            text=hover_cols,
            hovertemplate=hovertemplate_txt,
        )
    )

    fig.update_layout(
        xaxis={
            "title": xlabel,
            "tickmode": "array",
            "tickvals": [1, 1.5],
            "ticktext": [y1_label, y2_label],
        },
        yaxis={"title": ylabel},
        # showlegend=True,
        title=title,
    )

    # Show the figure
    fig.show()


def scatter_compare(
    df: pd.DataFrame,
    observed_label: str,
    reference_label: str = "",
    tolerance: float = 0.15,
    scatter_label: str = "",
    hovertext_columns: Optional[Union[list[pd.Series], list[str]]] = ["frame index"],
    plot_max: float = None,
    fig: go.Figure = None,
    show: bool = True,
):
    """A scatterplot for comparing the observed and reference data.

    i.e., correlation analysis, calculates the r2 between observed and reference,
    as well as the percentage of datapoints in the tolerance bounds of the reference
    data points. NaN values are ignored.

    Args:
        df (DataFrame): Dataframe containing the observed label and reference label
        observed_label (str): e.g., "reference"
        reference_label (str): e.g., "simple median"
        tolerance (float, optional): Marks the tolerance boundary on the scatter
            plot around the x=y line. If 0, neither it nor the 2x and 0.5x lines are shown. 
            Defaults to 0.15.
        scatter_label: Use the given label to label the scatter points. If not provided,
            then the observed label is used to label the scatter points.
        hovertext_columns: A list of pd.Series column names or a list of pd.Series;
            information in these columns will be added to the hover text.
        plot_max: Optional maximum value for x and y axes. Default: None (auto).
        fig (plotly graphics object): If a figure handle is given, we add new scatter
            points directly on top of the existing ones.
        show: if true, display the scatter plot.
    """
    df = df.dropna(subset=[reference_label, observed_label])
    x_data = df[reference_label]
    y_data = df[observed_label]
    if (len(x_data) == 0) or (len(y_data) == 0):
        print(f"scatter_compare() for `{scatter_label}`: The observed or the reference data argument was empty.")
        return fig, (None, None)

    if plot_max is None:
        plot_max = max(list(x_data) + list(y_data))

    # rounding for hover text display
    df.loc[:, reference_label] = np.round(df[reference_label], 2)
    df.loc[:, observed_label] = np.round(df[observed_label], 2)

    # Statistics
    obs_within_tolerance = (y_data >= x_data * (1 - tolerance)) & (
        y_data <= x_data * (1 + tolerance)
    )
    pct_in_tolerance = round(obs_within_tolerance.sum() / len(y_data) * 100, 2)
    r2 = np.round(r2_score(y_pred=y_data, y_true=x_data), 4)

    # Create the scatter plot
    overlay_flag = True
    if fig is None:
        overlay_flag = False
        fig = go.Figure()

    if scatter_label == "":
        scatter_label = observed_label

    hover_cols = df[["dataset-id", "frame index", reference_label, observed_label]]
    hovertemplate_txt = """session: %{text[0]}
                <br>frame: %{text[1]}
                <br>reference: %{text[2]:.2f}
                <br>observed:  %{text[3]:.2f}"""
    if hovertext_columns is not None:
        if type(hovertext_columns[0]) is str:
            for idx, col in enumerate(hovertext_columns):
                hover_cols = pd.concat([hover_cols, df[col]], axis=1)
                hovertemplate_txt += (f"<br>{col}") + ":%{text[" + str(idx + 3) + "]}"
        else:
            for idx, col in enumerate(hovertext_columns):
                hover_cols = pd.concat([hover_cols, col], axis=1)
                hovertemplate_txt += (
                    (f"<br>{col.name}") + ":%{text[" + str(idx + 3) + "]}"
                )

    if overlay_flag is False:
        # Add x=y line
        fig.add_trace(
            go.Scatter(
                x=[0, plot_max],
                y=[0, plot_max],
                mode="lines",
                line={"color": "white", "dash": "solid", "width":2},
                opacity=1,
                name="ideal line (x=y)",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker={"color":"navy", "opacity": 0.2, "size":5},
            name=scatter_label,
            text=hover_cols,
            hovertemplate=hovertemplate_txt,
        )
    )

    if overlay_flag is False:
        # Add lower tolerance boundary
        if tolerance:
            fig.add_trace(
                go.Scatter(
                    x=[0, plot_max],
                    y=[0, plot_max * (1 + tolerance)],
                    mode="lines",
                    line={"color": "black", "dash": "solid"},
                    name=f"Lower Tolerance ({tolerance*100}%)",
                )
            )

            # Add upper tolerance boundary
            fig.add_trace(
                go.Scatter(
                    x=[0, plot_max],
                    y=[0, plot_max * (1 - tolerance)],
                    mode="lines",
                    line={"color": "black", "dash": "solid"},
                    name=f"Upper Tolerance ({tolerance*100}%)",
                )
            )

            # Add y=2x line
            fig.add_trace(
                go.Scatter(
                    x=[0, plot_max],
                    y=[0, 2 * plot_max],
                    mode="lines",
                    line={"color": "#669099", "dash": "dash"},
                    name="2nd harmonic, y=2x",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, plot_max],
                    y=[0, 0.5 * plot_max],
                    mode="lines",
                    line={"color": "#669099", "dash": "dash"},
                    name="2nd harmonic, y=0.5*x",
                )
            )

        # Set axis labels and title
        title_text = f"Scatter plot of {observed_label} vs {reference_label}"
        if tolerance:
            sub_title = (
                f"{pct_in_tolerance}% of observed data lies in ± {tolerance*100}% of the reference data"  # noqa E501
                + f"<br>R2: {r2}"
            )  # noqa E501
        else:
            sub_title = ""    
        title_text_full = f"{title_text}<br><span style='font-size:12px'>{sub_title}</span>"
        fig.update_layout(
            xaxis_title = reference_label,
            yaxis_title = observed_label,
            title = title_text_full,
            yaxis = dict(
                range = [0, plot_max],
                constrain = 'domain'
            ),
            xaxis = dict(
                range = [0, plot_max],
                scaleanchor = 'y',    # Configure axes to have equal scale
                scaleratio = 1,
                constrain = 'domain',
                constraintoward = 'left'
            ),
            showlegend = True if tolerance else False,
            height = 600,
            width  = 600
        )

    # Display the plot
    if show:
        fig.show()

    return fig, (pct_in_tolerance, r2)


def scatter_with_marginal(
    df: pd.DataFrame,
    y_key: str,
    x_key: str,
    y_label: str = "y",
    x_label: str = "x",
    scatter_label: str = "",
    hovertext_columns: Optional[Union[list[pd.Series], list[str]]] = ["frame index"],
    title: str = "",
    sub_title: str = "",
    pt_color: str = "navy",
    show_linear: Optional[bool] = False,
    xbins_size: Optional[int] = None,
    ybins_size: Optional[int] = None,
    x_range: Optional[list] = None,
    y_range: Optional[list] = None,
    x_tick: Optional[int] = None,
    y_tick: Optional[int] = None,
    width: float = 400,
    height: float = 400,
    fig: go.Figure = None,
    showlegend: bool = False,
    show: bool = True,
    save_as: Optional[str] = None,
):
    """A scatterplot with linear regression and marginal distribution histograms.

    Args:
        df (DataFrame): Dataframe containing the x and y keys.
        y_key (str): e.g., "RR error"
        x_key (str): e.g., "ITA mean"
        y_label (str): For display, e.g., "RR error (bpm)"
        x_label (str): For display, e.g., "Mean ITA (°)"
        scatter_label: Use the given label to label the scatter points. If not provided,
            then the observed label is used to label the scatter points.
        hovertext_columns: A list of pd.Series column names or a list of pd.Series;
            information in these columns will be added to the hover text.
        title (str): Default: None, leading to an automatic summary
        sub_title (str): Default: None
        pt_color: CSS color of the scatter points. Default: navy.
        xbins_size: Size of bins for histogram. Default: None (auto)
        ybins_size: Size of bins for histogram. Default: None (auto).
        x_range: list of min and max. Default: None (auto).
        y_range: list of min and max. Default: None (auto).
        width: figure width. Default: 400.
        height: figure height. Default: 400.
        fig (plotly graphics object): If a figure handle is given, we add new scatter
            points directly on top of the existing ones.
        showlegend: If True, display the legend.
        show: If True, display the scatter plot.
        save_as: If set, the plot is saved as that path and name. A timestamp and .png 
            are appended. Default: None (not saved).
    Returns:
        A Plotly figure graph object
        r2 for the linear regression
    """
    df = df.dropna(subset=[x_key, y_key])
    x_data = df[x_key]
    y_data = df[y_key]
    if (len(x_data) == 0) or (len(y_data) == 0):
        print(f"scatter_linear_reg() for `{scatter_label}`: The observed or the reference data argument was empty.")
        return fig, (None, None)

    # rounding for hover text display
    df.loc[:, x_key] = np.round(df[x_key], 2)
    df.loc[:, y_key] = np.round(df[y_key], 2)

    histogram_fraction = 0.11
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[1-(histogram_fraction * height/width), (histogram_fraction * height/width)],
        row_heights=[histogram_fraction, 1-histogram_fraction],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05 * height/width,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
    )

    if scatter_label == "":
        scatter_label = y_label

    hover_cols = df[["dataset-id", "frame index", x_key, y_key]]
    hovertemplate_txt = """session: %{text[0]}
                <br>frame: %{text[1]}
                <br>x: %{text[2]:.2f}
                <br>y:  %{text[3]:.2f}"""
    if hovertext_columns is not None:
        if type(hovertext_columns[0]) is str:
            for idx, col in enumerate(hovertext_columns):
                hover_cols = pd.concat([hover_cols, df[col]], axis=1)
                hovertemplate_txt += (f"<br>{col}") + ":%{text[" + str(idx + 3) + "]}"
        else:
            for idx, col in enumerate(hovertext_columns):
                hover_cols = pd.concat([hover_cols, col], axis=1)
                hovertemplate_txt += (
                    (f"<br>{col.name}") + ":%{text[" + str(idx + 3) + "]}"
                )

    # Main data on scatterplot
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker={"color":pt_color, "opacity": 0.15, "size":4},
            name=scatter_label,
            text=hover_cols,
            hovertemplate=hovertemplate_txt,
        ),
        row=2, col=1
    )

    # Linear regression
    if show_linear:
        x_data_reshape = np.array(x_data).reshape((-1, 1))
        model = LinearRegression().fit(x_data_reshape, y_data)
        x_reg_plot = [np.min(x_data), np.max(x_data)]
        y_reg_plot = model.predict(np.array(x_reg_plot).reshape((-1, 1)))
        r2 = model.score(x_data_reshape, y_data)
        fig.add_trace(go.Scatter(
                x=x_reg_plot,
                y=y_reg_plot,
                name=f'linear regression',
                mode='lines',
                marker=dict(color='orange'),
            ), 
            row=2, col=1
        )
    else:  
        r2 = None

    # Right histogram of y distribution
    fig.add_trace(go.Histogram(
            y = y_data,
            xaxis = 'x2',
            ybins = dict(size=ybins_size) if ybins_size else None,
            marker = dict(
                color = 'rgba(192,192,192,1)'
            ),
            name = f'{y_label} count',
            hoverlabel = dict(namelength = -1),
            showlegend = False,
        ),
        row=2, col=2
    )
    # Top histogram of x distribution
    fig.add_trace(go.Histogram(
            x = x_data,
            yaxis = 'y2',
            xbins = dict(size=xbins_size) if xbins_size else None,
            marker = dict(
                color = 'rgba(192,192,192,1)'
            ),
            name = f'{x_label} count',
            hoverlabel = dict(namelength = -1),
            showlegend = False,
        ),
        row=1, col=1
    )

    # Set axis labels and title
    if not title:
        title = f"Scatter plot of {y_label} vs. {x_label}"
    if not sub_title:
        sub_title = f"n = {len(df)}"
    title_text_full = f"{title}<br><span style='font-size:12px'>{sub_title}</span>"
    fig.update_layout(
        title = title_text_full,
        showlegend = showlegend,
        height = height,
        width  = width,
    )
    fig.update_xaxes(title = x_label, row=2, col=1)
    fig.update_yaxes(title = y_label, row=2, col=1)
    if y_range:
        fig.update_yaxes(patch = dict(range = y_range), row=2, col=1)
    if x_range:
        fig.update_xaxes(patch = dict(range = x_range), row=2, col=1)
    if y_tick:
        fig.update_yaxes(dict(dtick = y_tick), row=2, col=1)
        fig.update_yaxes(dict(dtick = y_tick), row=2, col=2)
    if x_tick:
        fig.update_xaxes(dict(dtick = x_tick), row=2, col=1)
        fig.update_xaxes(dict(dtick = x_tick), row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=True,  row=2, col=2)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, showgrid=True,  row=1, col=1)

    # Display the plot
    if show:
        fig.show()

    # Save
    if save_as:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        fig.write_image(f"{save_as}_{current_time}.png", scale=4)

    return fig, r2


def plot_residuals(y_true: npt.NDArray, y_pred: npt.NDArray, title: str):
    """Generate a residuals plot. Y axis is expected-predicted, X axis is expected.

    Args:
        y_true: the expected value vector (i.e., reference value or target value).
        y_pred: the predicted value vector (i.e., output from a ML model).
        title (str): title of the plot (i.e., ML model description)

    Returns:
        None
    """
    residuals = y_true - y_pred

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=y_true, y=residuals, mode="markers"))
    fig.add_trace(
        go.Scatter(
            x=[min(y_true), max(y_true)], y=[0, 0], mode="lines", name="Zero line"
        )
    )
    fig.update_layout(
        title=f"Residuals Plot: {title}",
        xaxis_title="Target Values",
        yaxis_title="Residuals",
    )

    # Show the plot
    fig.show()

    return None


def plot_rr_to_hr_Nyquist(
    df, 
    HRkey, 
    RRtoHRkey, 
    RRkey, 
    minRRtoHRpertrial = 0, 
    thresholds = (0.4, 0.5), 
    opacity = 0.2, 
    showCountour = False, 
    bins = 10, 
    width = 800, 
    height = 800,
    save_as = None
    ):
    """
    Show how close the data were to the Nyquist rate. Rather than a scatter plot of RR vs. HR, we
    use a scatter of the RR-to-HR ratio vs. HR to make it easy to see the ratio and its
    distribution. We use RR/HR rather than the reverse to focus attention on high values being bad.

    Args:
        df: DataFrame for the entire trial. 
        HRkey, RRtoHRkey: Names of the keys in df to plot. Also displayed on axes. 
        RRkey: Name of the key in df that contains the RR (only used for isolines). 
        minRRtoHRpertrial: Exclude trials that don't contain a point that exceeds this.
        thresholds (tuple): Show lines at these levels with the percentage of points that 
            exceed them.
        opacity: 0-1. Opacity of markers.
        showCountour: True: Show a 2D contour of the distribution of points (can be confusing)
        bins: Number of bins for each of the histograms over each axis' entire range
        height, width: Size of the plot in pixels
        save_as: If set, the plot is saved as that path and name. A timestamp and .png 
            are appended. Default: None (not saved).

    Returns: None
    """
    # Set the upper limit of the displayed HR range
    HRmax = 240

    # Exclude any frames that lack our metrics
    df = df.dropna(subset=[HRkey, RRtoHRkey])

    num_frames_all = len(df[RRtoHRkey])
    if num_frames_all == 0:
        warnings.warn("plot_rr_to_hr_Nyquist(): No data to plot.")
        return

    x = df[HRkey]
    y = df[RRtoHRkey]

    fig = go.Figure()

    # Lines and annotations at key HR/RR thresholds
    
    for threshold in thresholds:
        num_frames_above = sum(df[RRtoHRkey] > threshold)
        pct_above = num_frames_above / num_frames_all

        fig.add_hline(
            y = threshold,
            line = {"color": "gray", "width": 1, "dash": "dot"},
            label = dict(
                text = f'↑{pct_above:6.1%}',
                textposition = 'start',
                yanchor = 'bottom',
                font={"size": 11, "color": "black"}
            ),
        )

    # Isolines for RR rates

    xfunc = np.arange(20, HRmax * 0.95, 1)
    RRmax = int(round(max(df[RRkey] + 3), -1)) + 10 # round up starting at 2, and add 10 in consideration of the for loop
    # RRmax = ceil(max(df[RRkey]) / 10) * 10          # round up to the nearest 10

    for RRval in range(10, RRmax, 10):
        yfunc = RRval / xfunc
        fig.add_trace(
            go.Scatter(
                x = xfunc,
                y = RRval / xfunc,
                xaxis = 'x',
                yaxis = 'y',
                mode = 'lines',
                line = {'color':'lightgray', 'width':1},
                name = f'RR {RRval:3d}',
                hoverinfo = 'skip',     # disable hover so that we can focus on the data
                showlegend = False,
            )
        )
        # Show the name at the right end of the trace that we just created
        fig.add_trace(
            go.Scatter(
                x = [fig.data[-1].x[-1]],
                y = [fig.data[-1].y[-1]],
                xaxis = 'x',
                yaxis = 'y',
                mode = 'text',
                text = f'RR {RRval:3d} ',
                textfont = dict(size=10, color='gray'),
                showlegend = False,
            )
        )

    # Primary data scatter, by trial, with color according to age
    # Make a color scale for the entire 0–90-year age range but with focus on < 5 yrs
    age_min = 0
    age_max = 90
    colorscale = [[0.00, 'rgb(255, 64, 64)'],
                  [0.05, 'rgb(210,210,  0)'],
                  [0.14, 'rgb(  0,  0,128)'],
                  [1.00, 'rgb(  0,  0,255)']
                  ]
    
    first_loop = True
    for dsid in df['dataset-id'].unique():
        filtered_df = df[df['dataset-id'] == dsid]
        
        if max(filtered_df[RRtoHRkey]) >= minRRtoHRpertrial:
            # skip plotting if that trial had no high values

            # For metadata, use the first row, since they all should be the same
            agetext =         filtered_df['subject age'].iloc[0]
            if agetext == '90+':     # mimic does this
                agenum = 90
                ageisnum = True
            elif agetext == '18-40':    # we do this for vortal, which doesn't provide individual ages
                agenum = 29            # vortal's median
                ageisnum = True
            elif agetext != agetext:    # a test for nan
                ageisnum = False
            else:
                try: 
                    agenum = float(agetext)
                    if agenum % 1 == 0:
                        agetext = int(agenum)
                    else:
                        agetext = round(agenum, 1)
                    ageisnum = True
                except ValueError:
                    ageisnum = False
            
            if 'ventilation' in filtered_df:
                ventilation = filtered_df['ventilation'].iloc[0]    
            else:
                ventilation = ''
            
            if ageisnum:
                colorthisseries = pc.sample_colorscale(colorscale, (agenum - age_min) / (age_max - age_min))[0]
            else:
                colorthisseries = 'gray'

            fig.add_trace(
                go.Scatter(
                    x = filtered_df[HRkey],
                    y = filtered_df[RRtoHRkey],
                    xaxis = 'x',
                    yaxis = 'y',
                    mode = 'markers',
                    name = f'{dsid}, age {agetext} y',
                    hoverlabel = dict(namelength = -1),
                    marker = dict(
                        color = colorthisseries,
                        cmin = age_min,
                        cmax = age_max,
                        colorscale = colorscale,
                        size = 3,
                        opacity = opacity
                    ),
                )
            )
            if first_loop:
                # Create the color bar just once, not every loop
                first_loop = False
                fig.update_traces(
                    marker = dict(
                        colorbar = dict(
                            title = 'Age (y)',
                            thickness = 10,
                            borderwidth = 0,
                            outlinewidth = 0,
                            len = 1.05,
                        ),
                    )
                )

    if showCountour:
        fig.add_trace(go.Histogram2dContour(
                x = x,
                y = y,
                colorscale = ['rgba(255, 255, 255, 0)', 'rgba(220, 220, 220, 0)'],
                # reversescale = True,
                showscale = False,
                xaxis = 'x',
                yaxis = 'y'
            ))

    # Right histogram of RR distribution
    hist_right = go.Histogram(
            y = y,
            xaxis = 'x2',
            ybins = dict(start=0, end=0.75, size=0.75/bins),
            marker = dict(
                color = 'rgba(192,192,192,1)'
            ),
            name = f'{HRkey} count',
            hoverlabel = dict(namelength = -1),
            showlegend = False
        )
    fig.add_trace(hist_right)

    # Top histogram of HR distribution
    hist_top = go.Histogram(
            x = x,
            yaxis = 'y2',
            xbins = dict(start=0, end=250, size=250/bins),
            marker = dict(
                color = 'rgba(192,192,192,1)'
            ),
            name = f'{RRtoHRkey} count',
            hoverlabel = dict(namelength = -1),
            showlegend = False
        )
    fig.add_trace(hist_top)

    numtrials = len(df['dataset-id'].unique())
    numdatasets = len(df['dataset'].unique())
    if numdatasets == 1:
        numdatasets_text = ''
    else:
        numdatasets_text = f'in {numdatasets} datasets'

    fig.update_layout(
        title = f"Nyquist check<br><span style='font-size:12px'>among {num_frames_all} frames from {numtrials} sessions {numdatasets_text}</span>",
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0,0.86],
            range = [0, HRmax],
            showgrid = True,
            title = HRkey,
            tickformat = '.0f'
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.84],
            # range = [min(y) * 0.9, max(max(y) * 1.1, 0.52)],     # set ymax to at least 0.52
            range = [0, 0.8],
            showgrid = True,
            title = RRtoHRkey,
            tickformat = '.1f',
            
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.91,1],
            nticks = 2,
            # range = [0, 60],
            showgrid = False,
            showticklabels = False,
            # title = 'Count'
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.89,1],
            nticks = 2,
            # range = [0, 60],
            showgrid = False,
            showticklabels = False,
            # title = 'Count'
        ),
        height = height,
        width = width,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False
    )

    fig.show()
    
    # Save
    if save_as:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        fig.write_image(f"{save_as}_{current_time}.png", scale=4)

    return None


def get_eval_metrics(
    df: pd.DataFrame,
    observed_label: str,
    reference_label: str,
    aggregate: bool = False,
    show: bool = False,
    as_df: bool = False,
):
    """Calls bland-altman and correlation analysis and print the resulting metrics.

    Correlation analysis calculates the pearson's correlation, while bland-altman
    analysis calculates the mean bias and limits of agreement for the dataset.

    In literature, MAE and RMSE are calculated for each subject. Then the median of
    these metrics are reported. If this is desired, set aggregate = True. I personally
    don't like to use these metrics, because they tend to overinflate the performance
    of the algorithm.

    Args:
        df: dataframe with observed_label and reference_label
        observed_label: column name in the dataframe.
        reference_label: column name in the dataframe.
        aggregate: if True, aggregate over the "trials" column via the mean, and 
            calculate the median MAE and median RMSE. Default is False.
        show: if true, show figures associated with evaluation metrics. Default false.
        as_df: if true, return output as a dataframe. if false (defualt), return as
            dictionary.

    Returns:
        results_dict, scatter_fig, ba_fig
    """
    if aggregate:
        df = df.groupby("trial").apply(
            lambda group: pd.Series(
                {
                    observed_label: group[observed_label].mean(),
                    reference_label: group[reference_label].mean(),
                    "MAE": np.abs(
                        group[reference_label] - group[observed_label]
                    ).mean(),
                    "RMSE": np.sqrt(
                        ((group[reference_label] - group[observed_label]) ** 2).mean()
                    ),  # noqa E501
                    "trial": np.round(group[reference_label].mean()),
                }
            )
        )
        hovertext_columns = None
        median_mean_abs_err = np.nanmedian(df["MAE"])
        median_rmse = np.nanmedian(df["RMSE"])
    else:
        hovertext_columns = ["frame index"]
        median_mean_abs_err = "n/a"
        median_rmse = "n/a"

    ba_fig, ba_metrics = bland_altman(
        df=df,
        reference=df[reference_label],
        observed=df[observed_label],
        reference_label=reference_label,
        observed_label=observed_label,
    )
    bias, loa_upper, loa_lower = ba_metrics

    scatter_fig, scatter_metrics = scatter_compare(
        df=df,
        reference_label=observed_label,
        observed_label=reference_label,
        hovertext_columns=hovertext_columns,
        show=show,
    )
    pct_in_tolerance, r2 = scatter_metrics

    results_dict = {
        "bias": bias,
        "loa": (loa_upper - loa_lower) / 2,
        "r2": r2,
        "pct in tolerance": pct_in_tolerance,
        "n": df.shape[0],
        "median MAE": median_mean_abs_err,
        "median RMSE": median_rmse,
    }

    if as_df:
        return (
            pd.DataFrame({key: [value] for key, value in results_dict.items()}),
            scatter_fig,
            ba_fig,
        )
    else:
        return results_dict, scatter_fig, ba_fig


def disagreement_vs_uncertainty(
    df_fig: pd.DataFrame,       # The dataset's results, usually filtered to exclude poor_quality_frames
    threshold=0,                # Dividing value of uncertainty for analyses of the subsets above and below
    height=800,                 # Plot height
    width=600,                  # Plot width
    title=""                    # Plot title
) -> None:
    """ Assesses the relationship between two frame-wide metrics:
    1) Median RR disagreement across the panel
    2) Mean uncertainty of the panel
    """

    # Drop any rows with missing data in our columns of interest
    df_fig = df_fig[['RR uncertainty panel (mean)', 'RR ref disagreement panel (bpm)', 'RR ref (mean)']].dropna()
    
    # Absolute disagreement

    x_data = df_fig['RR uncertainty panel (mean)']
    y_data_a = df_fig['RR ref disagreement panel (bpm)']

    x_for_reg = np.array(x_data).reshape((-1, 1))
    y_for_reg_a = np.array(y_data_a).reshape((-1, 1))
    model = LinearRegression().fit(x_for_reg, y_data_a)
    y_fit_a = model.predict(x_for_reg)
    r_sq_a = model.score(x_for_reg, y_for_reg_a)

    # Like above but as % disagreement

    y_data_p = df_fig['RR ref disagreement panel (bpm)'] / df_fig['RR ref (mean)']

    y_for_reg_p = np.array(y_data_p).reshape((-1, 1))
    model = LinearRegression().fit(x_for_reg, y_data_p)
    y_fit_p = model.predict(x_for_reg)
    r_sq_p = model.score(x_for_reg, y_for_reg_p)

    # Plot of both

    fig = make_subplots(rows=2, cols=1, subplot_titles=[f"Using absolute disagreement: r² = {r_sq_a:.2f}", f"Using percent disagreement: r² = {r_sq_p:.2f}"], shared_xaxes=True, vertical_spacing=0.07)

    fig.add_scatter(row=1, col=1, x = x_data, y = y_data_a, mode='markers', marker=dict(opacity=0.5))
    fig.add_scatter(row=1, col=1, x = x_data, y = y_fit_a,  mode='lines')
    fig.update_yaxes(row=1, col=1, title_text='RR ref disagreement panel (bpm)')

    fig.add_scatter(row=2, col=1, x = x_data, y = y_data_p, mode='markers', marker=dict(opacity=0.5))
    fig.add_scatter(row=2, col=1, x = x_data, y = y_fit_p,  mode='lines')
    fig.update_xaxes(row=2, col=1, title_text='RR uncertainty panel (mean)')
    fig.update_yaxes(row=2, col=1, title_text='% disagreement<BR>(RR ref disagreement panel / RR ref)', tickformat = '.0%')

    fig.update_layout(height=height, width=width, title=title, showlegend=False)

    fig.show()

    # Comparison above and below or equal to the uncertainty threshold

    x_below_idx = (x_data <= threshold)
    x_below = x_data[x_below_idx]
    y_a_below = y_data_a[x_below_idx]
    y_p_below = y_data_p[x_below_idx]

    x_above_idx = (x_data > threshold)
    x_above = x_data[x_above_idx]
    y_a_above = y_data_a[x_above_idx]
    y_p_above = y_data_p[x_above_idx]

    # Using escape codes for underlining: \033[4m and \033[0m
    print(f"Using an uncertainty threshold of {threshold}:")
    print()
    print(f"                                  \033[4mOverall\033[0m        \033[4mUncertainty ≤ thresh\033[0m   \033[4mUncertainty > thresh\033[0m")
    print(f"Fraction of all frames:            100.0%               {len(x_below) / len(x_data):6.1%}                 {len(x_above) / len(x_data):6.1%}")
    print(f"Mean disagreement:           {np.mean  (y_data_a):4.1f} bpm or {np.mean(  y_data_p):5.1%}    {np.mean  (y_a_below):4.1f} bpm or {np.mean(  y_p_below):5.1%}      {np.mean  (y_a_above):4.1f} bpm or {np.mean(  y_p_above):5.1%}")
    print(f"Median disagreement:         {np.median(y_data_a):4.1f} bpm or {np.median(y_data_p):5.1%}    {np.median(y_a_below):4.1f} bpm or {np.median(y_p_below):5.1%}      {np.median(y_a_above):4.1f} bpm or {np.median(y_p_above):5.1%}")
    print(f"Fraction with dis. > 3 bpm:        {len(y_data_a[y_data_a > 3]) / (len(y_data_a)+1e-6):6.1%}               {len(y_a_below[y_a_below > 3]) / (len(y_a_below)+1e-6):6.1%}                 {len(y_a_above[y_a_above > 3]) / (len(y_a_above)+1e-6):6.1%}")
    print(f"Fraction with dis. > 6 bpm:        {len(y_data_a[y_data_a > 6]) / (len(y_data_a)+1e-6):6.1%}               {len(y_a_below[y_a_below > 6]) / (len(y_a_below)+1e-6):6.1%}                 {len(y_a_above[y_a_above > 6]) / (len(y_a_above)+1e-6):6.1%}")

    left_out = len(x_data) - len(x_below) - len(x_above)
    if left_out:
        print()
        print(f"{left_out / len(x_data):.1%} of frames did not qualify for either side of the threshold.")

    return None