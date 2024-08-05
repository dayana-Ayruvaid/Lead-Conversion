import dash
from dash import Dash, html, dcc, Output, Input, callback, dash_table
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import re
import dash_bootstrap_components as dbc
import plotly.io as pio

# Defining app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# Reading DataFrame
df = pd.read_csv("lead_data.csv")

# Dropping Unnamed: 0
if "Unnamed: 0" in df.columns:
    df = df.drop(["Unnamed: 0"], axis=1)

# Lead Status replacement
df["Lead Status"] = df["Lead Status"].replace({
    "Won": "Appointment Fixed",
    "Appointment Fixed": "Appointment Fixed",
    "RNR": "RNR",
    "WIP": "WIP",
    "Lost": "Appointment Not Fixed"
})
df['utm_source_Campaign Source']=df['utm_source_Campaign Source'].replace(
    {"Google":"Google","google":"Google","fb":"Facebook","facebook":"Facebook","ig":"Instagram"}
)
variable = ["Unqualified", "WIP", "RNR", "General Enquiry", "Duplicate"]
df = df[~df["Lead Status"].isin(variable)]

columns = ['Gender', 'Age Group', 'Insured', 'Program Offering',
           'Severity', 'Chronicity', 'NFT Qualification', 'Spot', 'Lead Quality', 'utm_source_Campaign Source', 'Lead Source',
           'Primary Disease Specialty']

lead_status_options = df["Lead Status"].unique()

# Create a dark theme for plotly
pio.templates['custom_dark'] = go.layout.Template(
    layout=dict(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
)

app.layout = dbc.Container([
    html.Div([
        html.H1([
            html.Span("Lead Conversion"), html.Br(),
        ]),
        html.P("This dashboard prototype shows how to create an effective layout.")
    ],
        style={
            "vertical-alignment": "top",
            "height": 160
        }),
    html.Div(
        [
            dbc.RadioItems(
                id="view-selector",
                className='btn-group',
                inputClassName='btn-check',
                labelClassName="btn btn-outline-light",
                labelCheckedClassName="btn btn-light",
                options=[
                    {"label": "Data Summary", "value": "data_summary"},
                    {"label": "Graph", "value": "graph"},
                ],
                value="graph"
            ),

            html.Div(
                id="graph-container",
                children=[
                    dcc.Dropdown(columns, placeholder="Select a variable", id='variable-dropdown',style={'color': 'black','font-size': 20}, maxHeight=100, className='customDropdown'),
                    dcc.Dropdown(lead_status_options, placeholder="Select lead status",style={'color': 'black','font-size': 20}, id='status-dropdown', multi=True, maxHeight=100, className='customDropdown'),
                    dcc.Graph(id='bivariate-analysis'),
                    dash_table.DataTable(id='cross-table',
                                         style_header={'backgroundColor': 'black', 'color': 'white'},
                                         style_cell={'backgroundColor': 'black', 'color': 'white'},
                                         style_table={'overflowX': 'auto'}),
                    html.Button('See More', id='see-more-button', n_clicks=0),
                    html.Div(id='see-more-content', style={'display': 'none'})
                ],
                style={'display': 'block'}
            ),
            html.Div(
                id="data-summary-container",
                children=[
                    html.P("Data Summary will be here.")
                ],
                style={'display': 'none'}
            )
        ],
        style={
            'margin-left': 25,
            'margin-right': 25,
            'display': 'flex',
            'flex-direction': 'column'
        }
    )
], fluid=True, className='dashboard-container')


@app.callback(
    [Output('graph-container', 'style'),
     Output('data-summary-container', 'style')],
    [Input('view-selector', 'value')]
)
def toggle_view(selected_view):
    if selected_view == 'graph':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


@callback(
    [Output('bivariate-analysis', 'figure'),
     Output('cross-table', 'data'),
     Output('cross-table', 'columns')],
    [Input('variable-dropdown', 'value'),
     Input('status-dropdown', 'value')]
)
def update_output(variable, selected_statuses):
    print(f"Selected variable: {variable}")
    print(f"Selected statuses: {selected_statuses}")
    if variable and selected_statuses:
        filtered_df = df.loc[df["Lead Status"].isin(selected_statuses)]
        print(f"Filtered DataFrame:\n{filtered_df.head()}")
        cross_table = generate_cross_table(filtered_df, variable)
        table_data = cross_table.to_dict('records')
        table_columns = [{"name": i, "id": i} for i in cross_table.columns]
        bivariate_fig, overall_mean = bivariate_analysis(filtered_df, variable, cross_table, selected_statuses)
        #univariate_fig = univariate_analysis(filtered_df, variable)
        cross_table = add_population_mean(cross_table, overall_mean)
        table_data = cross_table.to_dict('records')
        return bivariate_fig,  table_data, table_columns
    return {}, [], []



@callback(
    Output('see-more-content', 'style'),
    Input('see-more-button', 'n_clicks')
)
def toggle_see_more(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}
    return {'display': 'none'}


def convert_age(age):
    age_str = str(age).lower()
    if 'yrs' in age_str or 'years' in age_str or 'year' in age_str:
        age_num = re.sub('[^0-9.]', '', age_str)
        return float(age_num) if age_num.replace('.', '').isdigit() else None
    elif 'months' in age_str or 'month' in age_str or 'm' in age_str:
        age_num = re.sub('[^0-9.]', '', age_str)
        return float(age_num) / 12 if age_num.replace('.', '').isdigit() else None
    elif 'weeks' in age_str or 'week' in age_str or 'w' in age_str:
        age_num = re.sub('[^0-9.]', '', age_str)
        return float(age_num) / 52 if age_num.replace('.', '').isdigit() else None
    elif 'days' in age_str or 'day' in age_str or 'd' in age_str:
        age_num = re.sub('[^0-9.]', '', age_str)
        return float(age_num) / 365 if age_num.replace('.', '').isdigit() else None
    else:
        age_num = re.sub('[^0-9.]', '', age_str)
        return float(age_num) if age_num.replace('.', '').isdigit() else None


df["Age"] = df["Age"].apply(convert_age)


def age_group_function(age):
    if age is None:
        return "Unknown"
    elif age >= 90:
        return '[90-100]'
    elif age >= 80:
        return '[80-90]'
    elif age >= 70:
        return '[70-80]'
    elif age >= 60:
        return '[60-70]'
    elif age >= 50:
        return '[50-60]'
    elif age >= 40:
        return '[40-50]'
    elif age >= 30:
        return '[30-40]'
    elif age >= 20:
        return '[20-30]'
    elif age >= 10:
        return '[10-20]'
    else:
        return '[0-10]'


df["Age Group"] = df["Age"].apply(age_group_function)
df = df[df["Age"] < 100]


def generate_cross_table(df, variable):
    # Treat missing values as a separate category
    df.loc[:, variable] = df[variable].fillna('Missing')
    cross_table = pd.crosstab(index=df[variable], columns=df['Lead Status'], margins=True, margins_name="Grand Total")

    # Reset index to move it into columns and add 'Sl No.'
    cross_table = cross_table.reset_index()
    cross_table['Sl No.'] = cross_table.index + 1

    total_fixed = cross_table["Appointment Fixed"].sum() if "Appointment Fixed" in cross_table.columns else 0
    total_not_fixed = cross_table["Appointment Not Fixed"].sum() if "Appointment Not Fixed" in cross_table.columns else 0

    if total_fixed > 0:
        cross_table['Appointment Fixed Percentage(%)'] = round((cross_table['Appointment Fixed'] / total_fixed) * 100, 0)
    else:
        cross_table['Appointment Fixed Percentage(%)'] = 0

    if total_not_fixed > 0:
        cross_table['Appointment Not Fixed Percentage(%)'] = round((cross_table['Appointment Not Fixed'] / total_not_fixed) * 100, 0)
    else:
        cross_table['Appointment Not Fixed Percentage(%)'] = 0

    # Reorder columns to have 'Sl No.' at the beginning
    cols = ['Sl No.'] + [col for col in cross_table.columns if col != 'Sl No.']
    cross_table = cross_table[cols]

    print(f"Cross Table:\n{cross_table.head()}")
    return cross_table


def add_population_mean(cross_table, overall_mean):
    cross_table['Population Mean'] = overall_mean
    return cross_table
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def bivariate_analysis(df, variable, cross_table, selected_statuses):
    # Exclude the "Total" and "Population Mean" rows for the plots
    cross_table_plot = cross_table[~cross_table[variable].isin(["Grand Total"])]
    
    # Extract counts and percentages from cross_table for the graphs
    appointment_fixed = cross_table_plot['Appointment Fixed'] if 'Appointment Fixed' in cross_table_plot.columns and 'Appointment Fixed' in selected_statuses else pd.Series()
    appointment_not_fixed = cross_table_plot['Appointment Not Fixed'] if 'Appointment Not Fixed' in cross_table_plot.columns and 'Appointment Not Fixed' in selected_statuses else pd.Series()

    population_percentage_fixed = cross_table_plot['Appointment Fixed Percentage(%)'] if 'Appointment Fixed Percentage(%)' in cross_table_plot.columns and 'Appointment Fixed' in selected_statuses else pd.Series()
    population_percentage_not_fixed = cross_table_plot['Appointment Not Fixed Percentage(%)'] if 'Appointment Not Fixed Percentage(%)' in cross_table_plot.columns and 'Appointment Not Fixed' in selected_statuses else pd.Series()

    if not appointment_fixed.empty and not appointment_not_fixed.empty:
        overall_mean = (appointment_fixed.sum() + appointment_not_fixed.sum()) / (len(appointment_fixed))
    elif not appointment_fixed.empty:
        overall_mean = appointment_fixed.mean()
    elif not appointment_not_fixed.empty:
        overall_mean = appointment_not_fixed.mean()
    else:
        overall_mean = 0

    # Ensure missing values are at the end
    bivariate_df = pd.DataFrame({
        variable: cross_table_plot[variable],
        "Count Fixed": appointment_fixed,
        "Count Not Fixed": appointment_not_fixed,
        "Population Percentage Fixed": population_percentage_fixed,
        "Population Percentage Not Fixed": population_percentage_not_fixed,
        "Total": appointment_fixed + appointment_not_fixed
    })

    # Sort with "Missing" at the end
    bivariate_df = pd.concat([
        bivariate_df[bivariate_df[variable] != "Missing"],
        bivariate_df[bivariate_df[variable] == "Missing"]
    ])

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, specs=[[{"secondary_y": True}]])

    if 'Appointment Fixed' in selected_statuses:
        fig.add_trace(
            go.Bar(
                x=bivariate_df[variable],
                y=bivariate_df["Count Fixed"],
                text=bivariate_df["Count Fixed"],
                textposition="inside",
                name="Count of Appointments Fixed",
                marker=dict(color="paleturquoise")
            ),
            row=1, col=1, secondary_y=False
        )

    if 'Appointment Not Fixed' in selected_statuses:
        fig.add_trace(
            go.Bar(
                x=bivariate_df[variable],
                y=bivariate_df["Count Not Fixed"],
                text=bivariate_df["Count Not Fixed"],
                textposition="inside",
                name="Count of Appointments Not Fixed",
                marker=dict(color="lightcoral")
            ),
            row=1, col=1, secondary_y=False
        )

    if not population_percentage_fixed.empty:
        fig.add_trace(
            go.Scatter(
                x=bivariate_df[variable],
                y=bivariate_df["Population Percentage Fixed"],
                text=[f"{v:.0f}%" for v in bivariate_df["Population Percentage Fixed"]],
                textposition="top center",
                name="Population Percentage Fixed",
                mode='lines+markers+text',
                marker=dict(color="blue", size=10),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )

    if not population_percentage_not_fixed.empty:
        fig.add_trace(
            go.Scatter(
                x=bivariate_df[variable],
                y=bivariate_df["Population Percentage Not Fixed"],
                text=[f"{v:.0f}%" for v in bivariate_df["Population Percentage Not Fixed"]],
                textposition="top center",
                name="Population Percentage Not Fixed",
                mode='lines+markers+text',
                marker=dict(color="green", size=10),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )

    max_total = bivariate_df["Total"].max() if not bivariate_df["Total"].empty else 0

    if overall_mean > 0:
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(bivariate_df) - 0.5,
            y0=overall_mean,
            y1=overall_mean,
            line=dict(color="Red", width=2),
            yref='y1',
            row=1, col=1
        )

        fig.add_annotation(
            x=len(bivariate_df),
            y=overall_mean,
            text=f"Overall Mean: {overall_mean:.2f}",
            showarrow=False,
            yref='y1',
            row=1, col=1
        )

    fig.update_layout(
        template='custom_dark',
        barmode='stack',
        title=f"Bivariate Analysis of Appointments by {variable}",
        xaxis_type='category',
        yaxis=dict(
            title="Count of Appointments",
            side="left",
            automargin=True,
            range=[0, max_total * 1.1]
        ),
        yaxis2=dict(
            title="Population Percentage",
            overlaying="y",
            side="right",
            ticksuffix="%",
            tickmode="sync",
            automargin=True,
        ),
        legend=dict(
            orientation="v",
            entrywidth=80,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.1
        ),
        height=600,
        width=1750,
        title_x=0.45
    )

    print(f"Figure data: {fig}")
    return fig, overall_mean



if __name__ == '__main__':
    app.run(debug=True)
