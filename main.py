from flask import Flask, render_template, request, redirect, url_for
import os
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
import json
import plotly
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.cluster import KMeans



app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Root URL
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')


# Get the uploaded files
@app.route("/", methods=['POST'])
def uploadFiles():
    for x in listdir('static/files/'):
        os.remove('static/files/'+ x)

    uploaded_file=request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
    return render_template('Dashboard.html')


@app.route("/tables", methods=['GET','POST'])
def tables():
    for x in listdir('static/files/'):

        df = pd.read_csv(x)
        df1 = df.head()
        df2 = pd.pivot_table(data=df, index='METER TYPE', columns='METER MONTH(NAME)', values='METER READING', aggfunc=np.sum)
        df3 = pd.pivot_table(data=df, index='METER TYPE', values='METER READING', aggfunc=np.sum)
        df2['Total'] = df3['METER READING']
        df2 = df2.sort_values(by=['Total'], ascending=False)
        df4 = pd.pivot_table(data=df, index='BP NUMBER', values='METER READING', columns='METER MONTH(NAME)', fill_value=0, aggfunc=np.sum)
        df5 = pd.pivot_table(data=df, index='BP NUMBER', values='METER READING', aggfunc=np.sum)
        df4['Total'] = df5['METER READING']
        df4 = df4.sort_values(by=['Total'], ascending=False)
        df6 = df4.head(15)
        df7 = df4.tail(15)

        return render_template('view.html', tables=[df1.to_html(classes='df1'), df2.to_html(classes = 'df2'), df6.to_html(classes= 'df6'), df7.to_html(classes = 'df7'), df4.to_html(classes = 'df4')],
                                    titles = ['na', 'First Five Rows of Dataset', 'METER TYPE wise and Month wise Consumptions', 'Top Ten Customer based on consumptions', 'Last Ten Customer based on consumptions', 'BP NUMBER and Month wise Consumptions'])


@app.route("/chart1", methods=['GET', 'POST'])
def chart1():
    for x in listdir('static/files/'):

        df = pd.read_csv(x)

        fig = px.bar(df, y='METER MONTH(NAME)', x='METER READING', color = 'METER TYPE')

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        header = "M-o-M Consumption Pattern"
        description = """
            Above bar graph showing month wise consumption pattern, with METER TYPE effect also showing with different color
            """
    return render_template('notdash2.html', graphJSON=graphJSON, header=header, description=description)


@app.route("/chart2", methods=['GET', 'POST'])
def chart2():
    for x in listdir('static/files/'):

        df = pd.read_csv(x)
        df['METER READING'] = df['METER READING'].fillna(0)
        df.loc[df['METER READING'] <= 0, 'missing reading'] = 0
        df.loc[df['METER READING'] > 0, 'missing reading'] = 1
        x = df.groupby('METER MONTH(NAME)')
        df1 = x['missing reading'].agg([len, np.sum])
        df1['%MISSING'] = (df1['sum'] / df1['len']-1) * 100
        df1.reset_index(inplace=True)

        fig1 = px.pie(df1, values=df1['len']- df1['sum'], names= 'METER MONTH(NAME)')
        plt.axis('equal')
        graphJSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        header = "M-o-M % Missing Reading"
        description = """
                If dataset having any missing reading details for any particulars month then showing here as pir charts form
                """
        return render_template('notdash2.html', graphJSON=graphJSON, header=header, description=description)


@app.route("/chart3", methods=['GET', 'POST'])
def chart3():
    for x in listdir('static/files/'):

        df = pd.read_csv(x)
        fig = go.Figure()

        BP_NUMBERS = list(df['BP NUMBER'].unique())[:20]

        for BP_NUMBER in BP_NUMBERS:
            fig.add_trace(
                go.Scatter(
                    x=df['METER MONTH(NAME)'][df['BP NUMBER'] == BP_NUMBER],
                    y=df['METER READING'][df['BP NUMBER'] == BP_NUMBER],
                    name=str(BP_NUMBER), visible=True
                )
            )

        buttons = []

        for i, BP_NUMBER in enumerate(BP_NUMBERS):
            args = [False] * len(BP_NUMBERS)
            args[i] = True

            button = dict(label=str(BP_NUMBER),
                          method="update",
                          args=[{"visible": args}])

            buttons.append(button)

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                type="dropdown",
                buttons=buttons,
                x=0,
                y=1.1,
                xanchor='left',
                yanchor='bottom'
            )],
            autosize=False,
            width=500,
            height=400
        )

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        header = "Customer wise M-o-M consumptions"
        description = """
                    Select customer number from dropdown and see the consumptions pattern of last 4 month
                    """
        return render_template('notdash2.html', graphJSON=graphJSON, header=header, description=description)


@app.route("/text", methods=['GET', 'POST'])
def text():
    for x in listdir('static/files/'):

        df = pd.read_csv(x)
        df2 = df.pivot_table(index='BP NUMBER', values='METER READING', columns='METER MONTH')
        df2.reset_index(inplace=True)
        BP_NUMBER = df2['BP NUMBER']
        df3 = df.pivot_table(index='BP NUMBER', values='METER READING', columns='METER MONTH')

        dfa = pd.DataFrame({8: [], 9: [],10: [], 11: []})

        for j in BP_NUMBER:

            for i in range(8, 11, 1):
                x = df3.loc[j, i]
                y = df3.loc[j, i + 1]
                if (x / y - 1) > 0.5 or (x / y - 1) < - 0.5:
                    f = df3.loc[j:j, :]
                    dfa = pd.concat([dfa, f], axis=0)

        return render_template('view.html', tables=[dfa.to_html(classes='dfa')], titles = ['na', '50% UP or DOWN consumptions from previsious month'])


@app.route("/chart4", methods=['GET', 'POST'])
def chart4():
    for x in listdir('static/files/'):

        df = pd.read_csv(x)
        df['Festival'] = df['Festival'].fillna('NO Festival')
        fig = px.bar(df, y='Festival', x='METER READING', color = 'METER MONTH(NAME)')

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        header = "Festival effect on M-o-M Consumption Pattern"
        description = """
            Above bar graph Festival effect on Month on Month consumption pattern
            """
        return render_template('notdash2.html', graphJSON=graphJSON, header=header, description=description)


@app.route("/cluster", methods=['GET', 'POST'])
def cluster():
    for x in listdir('static/files/'):

        df = pd.read_csv(x)
        df1 = df[['BP NUMBER', 'METER READING']]
        df3 = df1.pivot_table(index='BP NUMBER', values='METER READING', aggfunc=np.sum, sort=False).reset_index()
        scaled_data = scaler.fit_transform(df3)
        scaled_data = pd.DataFrame(scaled_data, columns=df3.columns)
        kmeans = KMeans(n_jobs=-1, n_clusters=7, init='k-means++')
        kmeans.fit(scaled_data)
        pred = kmeans.predict(scaled_data)
        centers = kmeans.cluster_centers_
        frame = pd.DataFrame(scaled_data)
        frame['cluster'] = pred
        frame['cluster'].value_counts()
        fig = px.scatter(frame, x='BP NUMBER', y='METER READING', color='cluster', opacity=0.8, size='cluster', size_max=30)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        header = "Clustering of the Customer based similar behavior. Nos of Clusters = 7"
        description = """
                consumer behavior on consumption and modelling out similar behavior consumer and show as cluster or circle graph
                
                Clustering of customer based on consumptions pattern by help of 
                unsupervised machine learning algorithm (KMeans clustering)
                nos of clusters: 7
                """
        return render_template('notdash2.html', graphJSON=graphJSON, header=header, description=description)

if (__name__ == "__main__"):
     app.run(port = 5000)