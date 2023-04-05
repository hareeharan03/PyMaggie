from flask import Flask, render_template, request, make_response, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
import sqlite3
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as pyo
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from scipy import stats
from sklearn.model_selection import train_test_split
import uuid
from datetime import datetime
import pymysql

import warnings

warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.", category=FutureWarning)


app = Flask(__name__, static_folder='staticFiles')

app.config['SECRET_KEY'] = 'Ha@03011999'
    
# create a connection to the database and create a table to store the output string
conn = pymysql.connect(host="localhost", user="root", password="Ha@03011999_PyMaggie", database="main_database")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS output_string (id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, session_id VARCHAR(255) NOT NULL UNIQUE, started_time TEXT, report_data TEXT)''')
conn.commit()

def output_info(inp):
        
    # check if session ID already exists in the output_string table
    c.execute("SELECT * FROM output_string WHERE session_id=%s AND started_time=%s", (session['session_id'],session['session_started_timestamp'],))
    result = c.fetchone()
    
    if result is None:
        # if no matching row exists, insert the session ID and input string as a new row
        c.execute("INSERT INTO output_string (session_id, started_time, report_data) VALUES (%s, %s, %s)", (session['session_id'], session['session_started_timestamp'], inp))
        print("thisss")
    else:
        # if matching row exists, append the input string to the existing report data
        report_data = str(result[3]) + str('<br><br>') + str(inp)
        c.execute("UPDATE output_string SET report_data=%s WHERE session_id=%s", (report_data, session['session_id'],))
        print("thattttt")
        
    conn.commit()

    



from google.cloud import storage
import os

def upload_csv_to_gcs(dataframe):
    """Uploads a CSV file to a specified Google Cloud Storage bucket in a specified folder.

    Args:
        csv_file_path (str): The local path to the CSV file to upload.
        folder_name (str): The name of the folder in the bucket to upload the file to.
    """
    # Instantiate a client for interacting with the Google Cloud Storage API

    from google.oauth2 import service_account

    # Replace [PATH_TO_PRIVATE_KEY_FILE] with the path to your private key file.
    credentials = service_account.Credentials.from_service_account_file('pymaggie-8cf90f8de8ce.json')

    client = storage.Client(credentials=credentials)

    # Get a reference to the specified bucket
    bucket = client.get_bucket("pymaggie_csv_storage")

    # Get a reference to the specified folder within the bucket
    folder = bucket.blob(str(session['session_id']))

    #changing into csv
    csv_file_path = str(session['session_id'])+"_uploaded"+'.csv'
    dataframe.to_csv(csv_file_path, index=False)

    # Get the name of the CSV file
    csv_file_name = os.path.basename(csv_file_path)

    # Uploads the CSV file to the destination folder
    blob = bucket.blob(str(session['session_id']) + "/" + csv_file_name)
    blob.upload_from_filename(csv_file_path)

    print(f"File {csv_file_name} uploaded to gs://pymaggie_csv_storage/{session['session_id']}/{csv_file_name}")







@app.route("/", methods=["GET", "POST"])
def home():
    session['session_id'] = str(uuid.uuid4())
    session['session_started_timestamp'] = str(datetime.now())

    global df

    if request.method == 'POST':
        print("YEs")
        file = request.files['file']

        submit_button=request.form.get("submit_button")
        use_sample_data=request.form.get("use_sample_data_button")
        
        if submit_button=="submitted":
            if file:
                # if file.content_length > 20 * 1024 * 1024:
                #     flash('File size exceeds limit of 20 MB', 'error')
                # elif not file.filename.endswith('.csv'):
                #     flash('File must be a CSV', 'error')
                # else:
                filename = secure_filename(file.filename)
                df = pd.read_csv(file)
                print(df.head())
                flash('File uploaded successfully', 'success')
                upload_csv_to_gcs(df)
                return redirect(url_for("initiating"))
        
        elif use_sample_data=="used-sample":
            df=pd.read_csv("application_record.csv")
            upload_csv_to_gcs(df)
            return redirect(url_for("initiating"))

    vare=False
    question ="can we start cooking the data?"

    
    return render_template("index.html", question=question, button_name="zero",list=False, upload=True,var=vare,show_welcome_popup=True)

@app.route("/initiating", methods=["GET", "POST"])
def initiating():

    question = "Do you want to perform data exploration?"
    message = ""

    if request.method == "POST":
        answer = request.form.get("first")
        #print("first",answer)
        if answer == "yes":
            return redirect(url_for("explore"))
        elif answer == "no":
            output_info("Proceeding to next step")
            return redirect(url_for("preprocessing"))
        
    session["previous_text"]=" "

    info="<b>What is data exploration?</b><br>&nbsp;&nbsp&nbsp;&nbsp;Data exploration is the process of analyzing and summarizing data to gain insights into its characteristics and properties. It involves visually inspecting and manipulating the data to identify patterns, trends, and relationships between variables. Data exploration is an essential step in the data analysis process and typically involves tasks such as data cleaning, descriptive statistics, data visualization, data transformation, and hypothesis testing. The overall goal of data exploration is to build a deeper understanding of the data and its characteristics."

    dataframe=(df.head()).to_html(header="true", table_id="df_table")
    
    return render_template("index.html", question=question, message=message, button_name="first",list=False,info=info,dataframe=dataframe)

@app.route("/explore", methods=["GET", "POST"])
def explore():

    if request.method == "POST":
        answerr = request.form.get("second")
        if answerr == "yes":
            return redirect(url_for("uniquevalues"))
        elif answerr == "no":
            return redirect(url_for("preprocessing"))

    #For printing column names
    df_columns = pd.DataFrame(columns=['Count','Column Name'])

    # Iterate over the column names and positions using enumerate and append them to df_positions
    for i, col_name in enumerate(df.columns):
        df_columns = df_columns.append({'Column Name': col_name, 'Count': int(i+1)}, ignore_index=True)

    html_table = df_columns.to_html(index=False,header="true", table_id="column_initial_table")


    global numerical,categorical
    categorical=list(df.select_dtypes(['object','bool']).columns)
    numerical=list(df.select_dtypes(['int64','float64']).columns)

    #For distinguish numerical and categorical columns
    temp_col=[]
    for i in numerical:
        if len(df[i].unique()) < 3:
            temp_col.append(i)
    categorical.extend(temp_col)
    numerical = [i for i in numerical if i not in temp_col]

    if len(categorical)!=len(numerical):
        diff=abs((len(categorical)) - (len(numerical)))
        if len(categorical) == (min(len(i) for i in [categorical, numerical])):
            for i in range(1,diff+1):
                categorical.append(" ")
        elif len(numerical) == (min(len(i) for i in [categorical, numerical])):
            for i in range(1,diff+1):
                numerical.append(" ")
    
    nested_list = [[x, y] for x, y in zip(categorical, numerical)]

    df_seperate = pd.DataFrame(nested_list, columns=['categorical Columns', 'Numerical Columns'])
    
    html_table_seperate = df_seperate.to_html(index=False,header="true", table_id="column_seperate_table")


    current_text_list=[("The shape of the given dataset {}".format(str(df.shape))),("The dataset has {} rows and {} columns".format(df.shape[0], df.shape[1])),
            (("These are the {} columns present in the dataset".format(str(df.shape[1])))),(html_table),("There are {} categorical columns and {} numerical columns".format(str(len(categorical)),str(len(numerical)))),
            (html_table_seperate)]
    
    current_text="<br><br>".join(current_text_list)
    
    #Here the previous input that has been stored to db is stored to session and checked whether the data is already stored or not to avoid duplicate
    if session.get('previous_text', None) != current_text:
        output_info(current_text)
        session['previous_text']=current_text

    
    # output_info("The shape of the given dataset {}".format(str(df.shape)))
    # output_info("The dataset has {} rows and {} columns".format(df.shape[0], df.shape[1]))
    # output_info(("These are the {} columns present in the dataset".format(str(df.shape[1]))))
    # output_info(html_table)
    # output_info("There are {} categorical columns and {} numerical columns".format(str(len(categorical)),str(len(numerical))))
    # output_info(html_table_seperate)

    question = "Want to analyse values in categorical column and numerical columns?"

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message, button_name="second",list=False,dataframe=dataframe)



@app.route("/uniquevalues", methods=["GET", "POST"])
def uniquevalues():
    global numerical_li

    if request.method == "POST":
        answerr = request.form.get("third")
        if answerr == "yes":
            return redirect(url_for("preprocessing_process"))
        elif answerr == "no":
            return redirect(url_for("direct_splitting_data"))

    #Removing empty string in the list
    categorical_li = [item for item in categorical if item != " "]
    numerical_li = [item for item in numerical if item != " "]

    plot_divs = []


    for col in categorical_li:
        unique_vals = df[col].value_counts()
        values = unique_vals.values.tolist()
        total = sum(values)
        percentages = [f"{(val/total)*100:.2f}%" for val in values]
        hover_text = [f"{val} ({perc})" for val, perc in zip(values, percentages)]
        trace = go.Bar(
            x=unique_vals.index.tolist(),
            y=unique_vals.values.tolist(),
            marker=dict(color=px.colors.qualitative.Pastel),
            hovertext=hover_text,
            hoverinfo='text'
        )
        layout = go.Layout(
            title=f"{col}",
            xaxis=dict(title="Value"),
            yaxis=dict(title="Count")
        )
        fig = go.Figure(data=[trace], layout=layout)
        plot_div = pyo.plot(fig, include_plotlyjs=True, output_type='div')
        plot_divs.append(plot_div)

    # create a list of HTML divs, each containing a plot_div for a categorical column
    divs = []
        
    divs.append('<div class="chart-grid">')
    for plot_div in plot_divs:
        divs.append(f'<div class="chart-cell">{plot_div}</div>')
    divs.append('</div>')
        
    # create a single HTML string by concatenating the HTML divs
    html = ''.join(divs)


    # create a list of HTML divs, each containing a plot_div for a categorical column
    divs = []
    
    divs.append('<div class="chart-grid">')
    for plot_div in plot_divs:
        divs.append(f'<div class="chart-cell">{plot_div}</div>')
    divs.append('</div>')

    # divs.append('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
    #create a single HTML string by concatenating the HTML divs
    html = ''.join(divs)

    #output_info(html)
    statistical_analysis=(df[(numerical_li)].describe())

    current_text=(statistical_analysis.to_html(header="true", table_id="table"))

    #Here the previous input that has been stored to db is stored to session and checked whether the data is already stored or not to avoid duplicate
    if session.get('previous_text', None) != current_text:
        output_info(current_text)
        session['previous_text']=current_text

    question="do you want to perform preprocessing?"

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message, button_name="third",list=False,dataframe=dataframe)

@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():

    if request.method == "POST":
        answerr = request.form.get("temp_third")
        if answerr == "yes":
            return redirect(url_for("preprocessing_process"))
        elif answerr == "no":
            return redirect(url_for("direct_splitting_data"))

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    question="do you want perform preprocessing?"
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message, button_name="temp_third",list=False,dataframe=dataframe)

@app.route("/preprocessing_process", methods=["GET", "POST"])
def preprocessing_process():

    if request.method == "POST":
        answerr = request.form.get("forth")
        if answerr == "yes":
            return redirect(url_for("null_values"))
        elif answerr == "no":
            return redirect(url_for("duplicate"))

     
    current_text="Preprocessing is on the process"

    #Here the previous input that has been stored to db is stored to session and checked whether the data is already stored or not to avoid duplicate
    if session.get('previous_text', None) != current_text:
        output_info(current_text)
        session['previous_text']=current_text

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    question="do you want treat null values?"
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message, button_name="forth",list=False,dataframe=dataframe)


@app.route("/null_values", methods=["GET", "POST"])
def null_values():

    #checking whether any columns has more number of null values
    dic_columns_with_more_null_values={}
    for i in list(df.columns):
        if ((df[i].isnull().sum())/len(df)) >= 0.2:
            dic_columns_with_more_null_values[str(i)]=((df[i].isnull().sum())/len(df))

    #if there is any column with more then 20% of null values when compared to total length of data this condition will be executed
    if len(dic_columns_with_more_null_values)>0:
        global null_value_columns
        null_value_columns=list(dic_columns_with_more_null_values.keys())
        suggestion=("SUGGESTION :You would have noticed that {} these columns contains more than 20% of null values. So, it is better to dropping these column which has more null values".format(list(dic_columns_with_more_null_values.keys())))
        question="Do you want to remove these columns to avoid data loss"

        #wait for user response for whether user wants to remove the column that has more null values 
        if request.method == "POST":
            answerr = request.form.get("fifth")

            #if yes redirect to route that removes the columns
            if answerr == "yes":
                return redirect(url_for("drop_null_columns"))
            elif answerr == "no":
                return redirect(url_for("drop_null_values"))


    else:
        question="Do you want to drop these null value rows"
        suggestion=" "

        if request.method == "POST":
            answerr = request.form.get("fifth")
            if answerr == "yes":
                return redirect(url_for("drop_null_values"))
            elif answerr == "no":
                return redirect(url_for("duplicate"))


    # output_info("These are the sum of null values in each columns of dataset")
    null_df = pd.DataFrame({'columns': (df.isnull().sum()).index, 'number of null values': (df.isnull().sum()).values})
    # output_info(null_df.to_html(header="true", table_id="null_value_table"))

    current_text_list=[("These are the sum of null values in each columns of dataset"),(null_df.to_html(header="true", table_id="null_value_table"))]
    current_text="<br><br>".join(current_text_list)
    
    if session.get('previous_text', None) != current_text:
        output_info(current_text)
        session['previous_text']=current_text

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message, button_name="fifth",suggestion=suggestion,list=False,dataframe=dataframe)

@app.route("/drop_null_columns", methods=["GET", "POST"])
def drop_null_columns():
    global df
    options = null_value_columns
    options.append("None_of_these")
    if request.method == "POST":
        # get the selected options from the form
        selected_options = request.form.getlist("options")
        
        #check if user selected none of these columns
        if "None_of_these" in selected_options:
            return redirect(url_for("drop_null_values"))
        else:
            df = df.drop(selected_options, axis=1)
            current_text=("{} columns has be droped from the dataset".format(selected_options))
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
            # output_info("{} columns has be droped from the dataset".format(selected_options))
            return redirect(url_for("drop_null_values"))

    question="Which of these columns would you like to drop?"

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=True,options=options,dataframe=dataframe)

@app.route("/drop_null_values", methods=["GET", "POST"])
def drop_null_values():
    global df

    if request.method == "POST":
        answerr = request.form.get("sixth")
        if answerr == "yes":
            df = (df.dropna())

            current_text=("In total {} rows has been dropped".format((str(df.isnull().values.sum()))))
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
            # output_info("In total {} rows has been dropped".format((str(df.isnull().values.sum()))))
            return redirect(url_for("duplicate"))
        elif answerr == "no":
            return redirect(url_for("duplicate"))

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    question="Do you want to remove null values"
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="sixth",dataframe=dataframe)

@app.route("/duplicate", methods=["GET", "POST"])
def duplicate():
    global df

    if request.method == "POST":
        answerr = request.form.get("seventh")
        if answerr == "yes":
            temp2=str("The dataset has {} duplicate rows".format((df.duplicated().sum())))
            # output_info(temp2)

            current_text=(temp2)
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
            
            return redirect(url_for("drop_duplicate_rows"))
        elif answerr == "no":
            return redirect(url_for("outliers"))


    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    question="Do you want to treat duplicate rows in the dataset?"
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="seventh",dataframe=dataframe)

@app.route("/drop_duplicate_rows", methods=["GET", "POST"])
def drop_duplicate_rows():
    global df,columns_with_more_unique_values
    columns_with_more_unique_values=[]
    for i in list(df.columns):
        if (df[i].nunique()/len(df)) > 0.8:
            columns_with_more_unique_values.append(i)
    print(columns_with_more_unique_values)
    if len(columns_with_more_unique_values) > 0:
        suggestion=("SUGGESTION : Sometime the dataset will have column which has more unique value such as customer number or application number etc. which will make every column as unique column even if other important features are same. Here the {} has more unique values so it is suggested to drop these kind of columns".format(columns_with_more_unique_values))
        question="Do you want to drop these columns?"
        if request.method == "POST":
            answerr = request.form.get("eigth")
            if answerr == "yes":
                return redirect(url_for("drop_columns_of_more_unique_values"))
            elif answerr == "no":
                return redirect(url_for("drop_duplicate_rows_values"))

    else:
        question="Do you want to drop the duplicate rows?"
        if int(df.duplicated().sum())!=0:
            suggestion=("SUGGESTION : The dataset has {} number of duplicates so it is advised to remove the duplicates".format(str(df.duplicated().sum())))
        if request.method == "POST":
            answerr = request.form.get("eigth")
            if answerr == "yes":
                current_text=("{} number of duplicate rows has been removed".format(str(df.duplicated().sum())))
                if session.get('previous_text', None) != current_text:
                    output_info(current_text)
                    session['previous_text']=current_text
                # output_info("{} number of duplicate rows has been removed".format(str(df.duplicated().sum())))
                df=df.drop_duplicates()
                return redirect(url_for("outliers"))
            elif answerr == "no":
                return redirect(url_for("outliers"))

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="eigth",suggestion=suggestion,dataframe=dataframe)

@app.route("/drop_columns_of_more_unique_values", methods=["GET", "POST"])
def drop_columns_of_more_unique_values():
    global df
    options = columns_with_more_unique_values
    options.append("None_of_these")

    if request.method == "POST":
        # get the selected options from the form
        selected_options = request.form.getlist("options")
        
        #check if user selected none of these columns
        if "None_of_these" in selected_options:
            return redirect(url_for("drop_duplicate_rows_values"))
        else:
            df = df.drop(selected_options, axis=1)
            
            current_text=("{} columns has be droped from the dataset".format(selected_options))
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
            # output_info("{} columns has be droped from the dataset".format(selected_options))

            return redirect(url_for("drop_duplicate_rows_values"))

    question="Which of these columns would you like to drop?"

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=True,options=options,dataframe=dataframe)

@app.route("/drop_duplicate_rows_values", methods=["GET", "POST"])
def drop_duplicate_rows_values():
    global df

    if request.method == "POST":
        answerr = request.form.get("ninth")
        if answerr == "yes":
            
            current_text=("{} number of duplicate rows has been removed".format(str(df.duplicated().sum())))
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
            # output_info("{} number of duplicate rows has been removed".format(str(df.duplicated().sum())))
            df=df.drop_duplicates()
            return redirect(url_for("outliers"))
        elif answerr == "no":
            return redirect(url_for("outliers"))
        
    df=df.drop_duplicates()

    current_text=("The duplicate rows has been removed")
    if session.get('previous_text', None) != current_text:
        output_info(current_text)
        session['previous_text']=current_text


    # output_info("The duplicate rows has been removed")

    question="Do you want to drop the duplicate rows?"

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="ninth",dataframe=dataframe)

@app.route("/outliers", methods=["GET", "POST"])
def outliers():

    if request.method == "POST":
        answerr = request.form.get("temp_ninth")
        if answerr == "yes":
            return redirect(url_for("remove_outliers"))
        elif answerr == "no":
            return redirect(url_for("encoding"))


    question="do you want to treat outliers?"

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="temp_ninth",dataframe=dataframe)

@app.route("/remove_outliers", methods=["GET", "POST"])
def remove_outliers():
    global df,numerical_columns
    
    temp_dataset=df
    numerical_columns=list(set(list(temp_dataset.columns)).intersection(numerical_li))
    for x in numerical_columns:
        q75,q25 = np.percentile(temp_dataset.loc[:,x],[75,25])
        intr_qr = q75-q25
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
        
        temp_dataset.loc[temp_dataset[x] < min,x] = np.nan
        temp_dataset.loc[temp_dataset[x] > max,x] = np.nan

    if request.method == "POST":
        answerr = request.form.get("tenth")
        if answerr == "yes":
            
            current_text=("In total {} rows has been dropped to remove outliers".format((str(df.isnull().values.sum()))))
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
                # output_info("In total {} rows has been dropped to remove outliers".format((str(df.isnull().values.sum()))))
            
            df = (temp_dataset.dropna())
            return redirect(url_for("encoding"))
        elif answerr == "no":
            return redirect(url_for("encoding"))
    
    outliers_df = pd.DataFrame({'columns': (temp_dataset.isnull().sum()).index, 'number of outliers': (temp_dataset.isnull().sum()).values})
    
    current_text=(outliers_df.to_html(header="true", table_id="outliers_table"))
    if session.get('previous_text', None) != current_text:
        output_info(current_text)
        session['previous_text']=current_text
    
    # output_info(outliers_df.to_html(header="true", table_id="outliers_table"))

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]
    question="Do you want to remove those outliers?"
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="tenth",dataframe=dataframe)

@app.route("/encoding", methods=["GET", "POST"])
def encoding():
    global df,cat_columns

    if request.method == "POST":
        answerr = request.form.get("eleventh")
        if answerr == "yes":
            return redirect(url_for("perform_encoding"))
        elif answerr == "no":
            return redirect(url_for("direct_splitting_data"))

    cat_columns = list(df.select_dtypes(['object','bool']).columns)
    if len(cat_columns) > 0:
        suggestion=("the dataset contains {} categrioal data columns which has to be encoded into numerical value".format(str(cat_columns)))
    else:
        suggestion=" "
    
    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    question="Do you want to perform encoding"
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="eleventh",suggestion=suggestion,dataframe=dataframe)

@app.route("/perform_encoding", methods=["GET", "POST"])
def perform_encoding():

    def check_length(lst):
        first_len = len(lst[0])
        for element in lst[1:]:
            if len(element) != first_len:
                return False
        return True

    global df,numerical_columns

    if request.method == "POST":
        answerr = request.form.get("twelth")
        if answerr == "yes":
            return redirect(url_for("splitting_data"))
        elif answerr == "no":
            return redirect(url_for("end_download_dataset"))

    for col in cat_columns:
        globals()['LE_{}'.format(col)] = LabelEncoder()
        df[col] = globals()['LE_{}'.format(col)].fit_transform(df[col])
    
    current_text=("These {} column has been encoded using Label encoding".format(cat_columns))
    if session.get('previous_text', None) != current_text:
        output_info(current_text)
        session['previous_text']=current_text
    # output_info("These {} column has been encoded using Label encoding".format(cat_columns))

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    question="Do you want to split the dataset into train and test?"
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="twelth",dataframe=dataframe)

@app.route("/direct_splitting_data", methods=["GET", "POST"])
def direct_splitting_data():

    global df

    if request.method == "POST":
        answerr = request.form.get("thirteen")
        if answerr == "yes":
            return redirect(url_for("splitting_data"))
        elif answerr == "no":
            return redirect(url_for("end_download_dataset"))
        
    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    question="Do you want to split the dataset into train and test?"
    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    return render_template("index.html", question=question, message=message,list=False,button_name="thirteen",dataframe=dataframe)


@app.route("/splitting_data", methods=["GET", "POST"])
def splitting_data():
    global df,predictor,target

    options = list(df.columns)
    if request.method == "POST":
        # get the selected options from the form
        selected_options = request.form.getlist("options")
        predictor=df.drop(columns=list(selected_options))
        target=df[selected_options]

        current_text=("The target(dependent) column is {} and predictor(independent) columns are {}".format(list(target.columns),list(predictor.columns)))
        if session.get('previous_text', None) != current_text:
            output_info(current_text)
            session['previous_text']=current_text
        # output_info("The target(dependent) column is {} and predictor(independent) columns are {}".format(list(target.columns),list(predictor.columns)))
        return redirect(url_for("select_splitting_type"))


    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    question="Please select the target variable!!"

    return render_template("index.html", question=question, message=message,list=False,dropdown=True,options=options,dataframe=dataframe)

@app.route("/select_splitting_type", methods=["GET", "POST"])
def select_splitting_type():
    global df,predictor,target

    options = ['stratified sampling', 'random sampling']

    if request.method == "POST":
        selected_option = request.form['options']

        if selected_option == "random sampling":
            return redirect(url_for("random_sample_splitting"))
        
        elif selected_option == "stratified sampling":
            return redirect(url_for("stratified_sample_splitting"))
        
    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    question="Please select the method that has to be used to split the data?"
    
    return render_template("index.html", question=question, message=message,list=False,dropdown=True,options=options,dataframe=dataframe)

@app.route("/random_sample_splitting", methods=["GET", "POST"])
def random_sample_splitting():
    global df,predictor,target,X_train, X_test, y_train, y_test

    options = ['70 : 30', '80 : 20']

    if request.method == "POST":
        selected_option = request.form['options']

        if selected_option == "70 : 30":

            current_text=("Random sampling is used to split the train and test data in 70 : 30 ratio")
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
            # output_info("Random sampling is used to split the train and test data in 70 : 30 ratio")

            X_train, X_test, y_train, y_test = train_test_split(predictor,target,test_size=0.3,random_state = 10)
            return redirect(url_for("end_download_dataset"))
        
        elif selected_option == "80 : 20":
            
            current_text=("Random sampling is used to split the train and test data in 80 : 20 ratio")
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
            output_info("Random sampling is used to split the train and test data in 80 : 20 ratio")

            X_train, X_test, y_train, y_test = train_test_split(predictor,target,test_size=0.2,random_state = 10)
            return redirect(url_for("end_download_dataset"))

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    question="Please select the method that has to be used to split the data?"

    return render_template("index.html", question=question, message=message,list=False,dropdown=True,options=options,dataframe=dataframe)

@app.route("/stratified_sample_splitting", methods=["GET", "POST"])
def stratified_sample_splitting():
    global df,predictor,target,X_train, X_test, y_train, y_test

    options = ['70 : 30', '80 : 20']

    if request.method == "POST":
        selected_option = request.form['options']

        if selected_option == "70 : 30":

            current_text=("stratified sampling is used to split the train and test data in 70 : 30 ratio")
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
            # output_info("stratified sampling is used to split the train and test data in 70 : 30 ratio")

            X_train, X_test, y_train, y_test = train_test_split(predictor,target, test_size=0.3, random_state=0, stratify=target)
            return redirect(url_for("end_download_dataset"))
        
        elif selected_option == "80 : 20":

            current_text=("stratified sampling is used to split the train and test data in 80 : 20 ratio")
            if session.get('previous_text', None) != current_text:
                output_info(current_text)
                session['previous_text']=current_text
                # output_info("stratified sampling is used to split the train and test data in 80 : 20 ratio")
            
            X_train, X_test, y_train, y_test = train_test_split(predictor,target, test_size=0.2, random_state=0, stratify=target)
            return redirect(url_for("end_download_dataset"))

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    question="Please select the method that has to be used to split the data?"

    return render_template("index.html", question=question, message=message,list=False,dropdown=True,options=options,dataframe=dataframe)

@app.route("/end_download_dataset", methods=["GET", "POST"])
def end_download_dataset():

    global df,predictor,target,X_train, X_test, y_train, y_test

    output_info("Thank you for using PyMaggie")

    c.execute("SELECT report_data FROM output_string WHERE session_id=%s", (session['session_id'],))
    result = c.fetchone()
    if result:
        message = result[0]

    dataframe=(df.head()).to_html(header="true", table_id="df_table")

    question="Please select the method that has to be used to split the data?"

    return render_template("index.html", question=question, message=message,list=False,dropdown=False,dataframe=dataframe)

# @app.route("/feature_scaling", methods=["GET", "POST"])
# def feature_scaling():
#     global df,temp2

#     def check_length(lst):
#         first_len = len(lst[0])
#         for element in lst[1:]:
#             if len(element) != first_len:
#                 return False
#         return True
    
#     if request.method == "POST":
#         answerr = request.form.get("eleventh")
#         if answerr == "yes":
#             return redirect(url_for("select_feature_scaling"))
#         elif answerr == "no":
#             return redirect(url_for("uniquevalues"))
    
#     #it is to filter the numerical columns that are more then 2 unique values
#     temp=list(df.select_dtypes(['int64','float64']).columns)
#     temp2=[]
#     for i in temp:
#         if len(df[i].unique()) > 2:
#             temp2.append(i)

#     #will get the mean value of each column present in temp2
#     mean_value_of_each_columns={}
#     for i in temp2:
#         mean_value_of_each_columns[i]=str(int(round((df[[i]].mean()[0]),3)))

#     #checking whether all colum mean values are in same scale
#     result_FS=check_length(list(mean_value_of_each_columns.values()))

#     if result_FS == False:
#         suggestion="suggestion : It is adviced to perform feature scaling since columns in dataset are in different scale"
#     else:
#         suggestion=" "

#     c.execute("SELECT string FROM output_string")
#     rows = c.fetchall()
#     message =  '<br><br>'.join([str(row[0]) for row in rows])

#     question="Do you want to perform feature scaling"

#     return render_template("index.html", question=question, message=message,list=False,button_name="eleventh",suggestion=suggestion)

# @app.route("/select_feature_scaling", methods=["GET", "POST"])
# def select_feature_scaling():
#     global df
#     options = ['Normalization', 'Standardization']

#     if request.method == "POST":
#         selected_option = request.form['options']

#         if selected_option == "Normalization":
#             for column in temp2:
#                 df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min()) 
#                 output_info("These {} columns has been scaled using the normalization technique".format(temp2))
#             return redirect(url_for("encoding"))
        
#         elif selected_option == "Standardization":
#             for column in temp2:
#                 df[column] = stats.zscore(df[column])
#             output_info("These {} columns has been scaled using the Standardization technique".format(temp2))
#             return redirect(url_for("encoding"))

#     question="which technique do you want to perform?"

#     c.execute("SELECT string FROM output_string")
#     rows = c.fetchall()
#     message =  '<br><br>'.join([str(row[0]) for row in rows])

#     return render_template("index.html", question=question, message=message,list=False,button_name="eleventh",dropdown=True,options=options)




if __name__ == "__main__":
    app.run(debug=True, threaded=True)