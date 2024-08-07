import dash
from dash import Dash, html, dcc, Output, Input, callback, dash_table
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import re
import dash_bootstrap_components as dbc
import plotly.io as pio

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# Load the data
df = pd.read_csv("lead_data.csv")
print("Initial data shape:", df.shape)

if "Unnamed: 0" in df.columns:
    df = df.drop(["Unnamed: 0"], axis=1)
print("After dropping 'Unnamed: 0' column:", df.shape)

# Replace values in 'Lead Status'
df["Lead Status"] = df["Lead Status"].replace({
    "Won": "Appointment Fixed",
    "Appointment Fixed": "Appointment Fixed",
    "RNR": "RNR",
    "WIP": "WIP",
    "Lost": "Appointment Not Fixed"
})
df['utm_source_Campaign Source'] = df['utm_source_Campaign Source'].replace({
    "Google": "Google",
    "google": "Google",
    "fb": "Facebook",
    "facebook": "Facebook",
    "ig": "Instagram"
})

# Filter out unwanted lead statuses
variable = ["Unqualified", "WIP", "RNR", "General Enquiry", "Duplicate"]
df = df[~df["Lead Status"].isin(variable)]
print("After filtering 'Lead Status':", df.shape)

# Convert age
def convert_age(age):
    age_str = str(age).lower().strip()
    if re.match(r'^\d+\.?\d*$', age_str):  # Handle purely numeric ages
        return float(age_str)
    elif 'yrs' in age_str or 'years' in age_str or 'year' in age_str:
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
        return None

df["Age"] = df["Age"].apply(convert_age)
print("After converting age:", df.shape)
print("Unique ages after conversion:", df["Age"].unique())  # Debug print

# Create age groups including "Unknown" for missing/invalid ages
def age_group_function(age):
    if pd.isna(age):
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
print("After creating age groups:", df.shape)

# Print age group distribution for debugging
print("Age group distribution:\n", df["Age Group"].value_counts())  # Debug print

columns = ['Gender', 'Age Group', 'Insured', 'Program Offering', 'Severity', 'Chronicity', 'NFT Qualification', 'Spot', 'Lead Quality', 'utm_source_Campaign Source', 'Lead Source',
           'Primary Disease Specialty']

lead_status_options = [status for status in df["Lead Status"].unique() if pd.notnull(status)]

pio.templates['custom_dark'] = go.layout.Template(layout=dict(paper_bgcolor='black', plot_bgcolor='black', font=dict(color='white')))

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
                    dcc.Dropdown(columns, placeholder="Select a primary variable", id='primary-variable-dropdown', style={'color': 'black', 'font-size': 20}, maxHeight=100, className='customDropdown'),
                    dcc.Dropdown(columns, placeholder="Select secondary variables", style={'color': 'black', 'font-size': 20}, id='secondary-variable-dropdown', multi=True, maxHeight=100, className='customDropdown'),
                    dcc.Dropdown(lead_status_options, placeholder="Select lead status", style={'color': 'black', 'font-size': 20}, id='status-dropdown', multi=True, maxHeight=100, className='customDropdown'),
                    dcc.Graph(id='multivariate-analysis'),
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
    )], fluid=True, className='dashboard-container')

@app.callback([Output('graph-container', 'style'), Output('data-summary-container', 'style')], [Input('view-selector', 'value')])
def toggle_view(selected_view):
    if selected_view == 'graph':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

@callback(
    [Output('multivariate-analysis', 'figure'),
     Output('cross-table', 'data'),
     Output('cross-table', 'columns')],
    [Input('primary-variable-dropdown', 'value'),
     Input('secondary-variable-dropdown', 'value'),
     Input('status-dropdown', 'value')]
)
def update_output(primary_variable, secondary_variables, selected_statuses):
    if primary_variable and selected_statuses:
        filtered_df = df.loc[df["Lead Status"].isin(selected_statuses)]
        print("Filtered dataframe shape:", filtered_df.shape)  # Debug print
        cross_table = generate_cross_table(filtered_df, primary_variable, secondary_variables)
        table_data = cross_table.to_dict('records')
        table_columns = [{"name": i, "id": i} for i in cross_table.columns]
        multivariate_fig = multivariate_analysis(filtered_df, primary_variable, secondary_variables, cross_table, selected_statuses)
        return multivariate_fig, table_data, table_columns
    return {}, [], []

@callback(Output('see-more-content', 'style'), Input('see-more-button', 'n_clicks'))
def toggle_see_more(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}
    return {'display': 'none'}

def generate_cross_table(df, primary_variable, secondary_variables):
    if secondary_variables:
        variables = [primary_variable] + secondary_variables
        df = df.fillna('Missing')
        cross_table = pd.crosstab(index=[df[var] for var in variables], columns=df['Lead Status'], margins=True, margins_name="Grand Total")
    else:
        df[primary_variable] = df[primary_variable].fillna('Missing')
        cross_table = pd.crosstab(index=df[primary_variable], columns=df['Lead Status'], margins=True, margins_name="Grand Total")

    cross_table = cross_table.reset_index()
    cross_table['Sl No.'] = cross_table.index + 1

    total_fixed = cross_table.at[cross_table.index[-1], "Appointment Fixed"] if "Appointment Fixed" in cross_table.columns else 0
    total_not_fixed = cross_table.at[cross_table.index[-1], "Appointment Not Fixed"] if "Appointment Not Fixed" in cross_table.columns else 0

    if total_fixed > 0:
        cross_table['Appointment Fixed Percentage(%)'] = round((cross_table['Appointment Fixed'] / total_fixed) * 100, 0)
    else:
        cross_table['Appointment Fixed Percentage(%)'] = 0

    if total_not_fixed > 0:
        cross_table['Appointment Not Fixed Percentage(%)'] = round((cross_table['Appointment Not Fixed'] / total_not_fixed) * 100, 0)
    else:
        cross_table['Appointment Not Fixed Percentage(%)'] = 0

    cols = ['Sl No.'] + [col for col in cross_table.columns if col != 'Sl No.']
    cross_table = cross_table[cols]
    
    print("Generated cross table shape:", cross_table.shape)  # Debug print

    return cross_table

def multivariate_analysis(df, primary_variable, secondary_variables, cross_table, selected_statuses):
    cross_table_plot = cross_table[~cross_table[primary_variable].isin(["Grand Total"])]
    appointment_fixed = cross_table_plot['Appointment Fixed'] if 'Appointment Fixed' in cross_table_plot.columns else pd.Series()
    appointment_not_fixed = cross_table_plot['Appointment Not Fixed'] if 'Appointment Not Fixed' in cross_table_plot.columns else pd.Series()

    population_percentage_fixed = cross_table_plot['Appointment Fixed Percentage(%)'] if 'Appointment Fixed Percentage(%)' in cross_table_plot.columns else pd.Series()
    population_percentage_not_fixed = cross_table_plot['Appointment Not Fixed Percentage(%)'] if 'Appointment Not Fixed Percentage(%)' in cross_table_plot.columns else pd.Series()

    # Calculate the overall mean for age groups
    if primary_variable == "Age Group":
        num_buckets = len(df["Age Group"].unique())
        overall_mean = (appointment_fixed.sum() + appointment_not_fixed.sum()) / num_buckets
    else:
        overall_mean = (appointment_fixed.sum() + appointment_not_fixed.sum()) / (len(appointment_fixed))

    bivariate_df = pd.DataFrame({
        primary_variable: cross_table_plot[primary_variable],
        "Count Fixed": appointment_fixed,
        "Count Not Fixed": appointment_not_fixed,
        "Population Percentage Fixed": population_percentage_fixed,
        "Population Percentage Not Fixed": population_percentage_not_fixed,
        "Total": appointment_fixed + appointment_not_fixed
    })
    bivariate_df = pd.concat([
        bivariate_df[bivariate_df[primary_variable] != "Missing"],
        bivariate_df[bivariate_df[primary_variable] == "Missing"]])

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1, specs=[[{"secondary_y": True}]])

    if 'Appointment Fixed' in selected_statuses:
        fig.add_trace(
            go.Bar(
                x=bivariate_df[primary_variable],
                y=bivariate_df["Count Fixed"],
                text=bivariate_df["Count Fixed"],
                textposition="inside",
                name="Count of leads Fixed Appointments",
                marker=dict(color="paleturquoise")
            ),
            row=1, col=1, secondary_y=False
        )

    if 'Appointment Not Fixed' in selected_statuses:
        fig.add_trace(
            go.Bar(
                x=bivariate_df[primary_variable],
                y=bivariate_df["Count Not Fixed"],
                text=bivariate_df["Count Not Fixed"],
                textposition="inside",
                name="Count of leads not fixed Appointments",
                marker=dict(color="lightcoral")
            ),
            row=1, col=1, secondary_y=False
        )

    if not population_percentage_fixed.empty:
        fig.add_trace(
            go.Scatter(
                x=bivariate_df[primary_variable],
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
                x=bivariate_df[primary_variable],
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

    # Only draw the overall mean if both "Appointment Fixed" and "Appointment Not Fixed" are selected
    if 'Appointment Fixed' in selected_statuses and 'Appointment Not Fixed' in selected_statuses:
        fig.add_shape(type="line", x0=-0.5, x1=len(bivariate_df) - 0.5, y0=overall_mean, y1=overall_mean, line=dict(color='firebrick', width=4, dash='dot'), yref='y1', row=1, col=1)
        fig.add_annotation(
            x=len(bivariate_df) - 0.5,
            y=overall_mean,
            text=f"Population Mean: {overall_mean:.2f}",
            showarrow=False,
            yref='y1',
            row=1, col=1
        )

    if secondary_variables:
        for sec_var in secondary_variables:
            secondary_group = df.groupby([primary_variable, sec_var]).size().reset_index(name='Count')
            secondary_group_pivot = secondary_group.pivot(index=primary_variable, columns=sec_var, values='Count').fillna(0)
            secondary_group_pivot = secondary_group_pivot.reindex(bivariate_df[primary_variable])
            
            for col in secondary_group_pivot.columns:
                secondary_group_pivot[col] = (secondary_group_pivot[col] / secondary_group_pivot.sum(axis=1)) * 100

            for col in secondary_group_pivot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=secondary_group_pivot.index,
                        y=secondary_group_pivot[col],
                        text=[f"{v:.0f}%" for v in secondary_group_pivot[col]],
                        textposition="top center",
                        name=f"{sec_var}: {col}",
                        mode='lines+markers+text',
                        marker=dict(size=10),
                        yaxis='y2'
                    ),
                    row=1, col=1, secondary_y=True
                )

    fig.update_layout(
        template='custom_dark',
        barmode='stack',
        title=f"Analysis of Appointments by {primary_variable}",
        xaxis_type='category',
        yaxis=dict(
            title="Count of leads",
            side="left",
            automargin=True,
            range=[0, max_total * 1.1]
        ),
        yaxis2=dict(
            title="Percentage(%)",
            overlaying="y",
            side="right",
            ticksuffix="%",
            tickmode="sync",
            automargin=True,
            range=[0, 100]
        ),
        legend=dict(orientation="v", entrywidth=80, yanchor="bottom", y=1.02, xanchor="right", x=1.1),
        height=600, width=2000, title_x=0.45)

    return fig


if __name__ == '__main__':
    app.run(debug=True)
