# from flask import Flask, render_template, request, make_response
# import pandas as pd
# import time
# app = Flask(__name__)


# def get_question(question_number):
#     if question_number == 1:
#         return "Do you like ice creams?"
#     elif question_number == 2:
#         return "Which flavor ice cream do you like?"

# @app.route("/", methods=["GET", "POST"])
# def home():
#     df=pd.read_csv("application_record.csv")
    
#     question="1. do you want to perform data exploration"
#     message=""
#     #print(request.form.getlist)
#     if request.method == "POST":
        
#         answer = request.form.get("answer")
#         print(request.form.get('question'))
#         if answer == "yes":
#             print('Hello')
#             message = str("The shape of the given dataset{}".format(str(df.shape)))
#             question="2. do you want to see values"
#             #render_template("index.html", question=question, message=message)
#         elif answer=='yes' and question=="do you want to see values":
#             message = 'New message'
#             question="3. New question"
#         else:
#             message="processing to next step"
#             question="do you want to perform preprocessing"
#             #return render_template("index.html", question="do you want to see uniquew values", message=message)

    
#     return render_template("index.html", question=question, message=message)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, make_response, redirect, url_for
import pandas as pd
import re
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as pyo

app = Flask(__name__, static_folder='staticFiles')

df = pd.read_csv("application_record.csv")

output_string=""
def output_info(inp):
    global output_string
    #recent_string = re.sub(r"\n", "<br>", str(inp))
    output_string=output_string+"<br><br>"+inp
    return None
    

    

@app.route("/", methods=["GET", "POST"])
def home():
    output_string=""
    question = "Do you want to perform data exploration?"
    message = ""
    perform_exploration = None

    if request.method == "POST":
        print("first",request.form.get("first"))
        answer = request.form.get("first")
        #print("first",answer)
        if answer == "yes":
            return redirect(url_for("explore"))
        elif answer == "no":
            message = "Proceeding to next step"
            question = "Do you want to perform preprocessing?"
            return render_template("index.html", question=question, message=message,list=False)

    return render_template("index.html", question=question, message=message, button_name="first",list=False)

@app.route("/explore", methods=["GET", "POST"])
def explore():

    if request.method == "POST":
        answerr = request.form.get("second")
        if answerr == "yes":
            return redirect(url_for("uniquevalues"))
        elif answerr == "no":
            question="do you want perform preprocessing"
            return redirect(url_for("uniquevalues"))

    output_info("The shape of the given dataset {}".format(str(df.shape)))
    output_info("The dataset has {} rows and {} columns".format(df.shape[0], df.shape[1]))
    output_info(("These are the {} columns present in the dataset".format(str(df.shape[1]))))

    #For printing column names
    ind=1
    temp=[]
    for i in df.columns:
        temp.append((str(ind)+("  "*(3-(len(str(ind)))))+i))
        ind=ind+1
    output_info('<br>'.join(temp))

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
    output_info("There are {} categorical columns and {} numerical columns".format(str(len(categorical)),str(len(numerical))))

    if len(categorical)!=len(numerical):
        diff=abs((len(categorical)) - (len(numerical)))
        if len(categorical) == (min(len(i) for i in [categorical, numerical])):
            for i in range(1,diff+1):
                categorical.append(" ")
        elif len(numerical) == (min(len(i) for i in [categorical, numerical])):
            for i in range(1,diff+1):
                numerical.append(" ")
    
    nested_list = [[x, y] for x, y in zip(categorical, numerical)]

    temp_table=[]
    for names in nested_list:
        temp_table.append("<tr><td>{}</td><td>{}</td></tr>".format(names[0],names[1]))
    table="<table><thead><tr><th>Categorial Columns</th><th>Numerical Columns</th></tr></thead><tbody>"+(''.join(temp_table))+"</tbody></table>"
    output_info(table)
    
    message = output_string
    question = "Want to analyse values in categorical column and numerical columns?"

    return render_template("index.html", question=question, message=message, button_name="second",list=False)



@app.route("/uniquevalues", methods=["GET", "POST"])
def uniquevalues():

    if request.method == "POST":
        answerr = request.form.get("third")
        if answerr == "yes":
            return redirect(url_for("preprocessing"))
        elif answerr == "no":
            question="do you really dont wanna perform preprocessing"
            return redirect(url_for("uniquevalues"))

    #Removing empty string in the list
    categorical_li = [item for item in categorical if item != ""]
    numerical_li = [item for item in numerical if item != " "]

    plot_divs = []

    # loop through each categorical column and create a plot_div for it
    # for col in categorical_li:
    #     unique_vals = df[col].value_counts()
    #     trace = go.Bar(x=unique_vals.index.tolist(), y=unique_vals.values.tolist(),marker=dict(color=px.colors.qualitative.Pastel))
    #     layout = go.Layout(
    #         title=f"{col}",
    #         xaxis=dict(title="Value"),
    #         yaxis=dict(title="Count")
    #     )
    #     fig = go.Figure(data=[trace], layout=layout)
    #     plot_div = pyo.plot(fig, include_plotlyjs=True, output_type='div')
    #     plot_divs.append(plot_div)

    # loop through each categorical column and create a plot_div for it
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

    output_info(html)
    statistical_analysis=(df[(numerical_li)].describe())
    output_info(statistical_analysis.to_html(header="true", table_id="table"))

    question="do you want to perform preprocessing?"
    message = output_string
    return render_template("index.html", question=question, message=message, button_name="third",list=False)


@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():

    if request.method == "POST":
        answerr = request.form.get("forth")
        if answerr == "yes":
            return redirect(url_for("null_values"))
        elif answerr == "no":
            return redirect(url_for("uniquevalues"))

    output_info("Preprocessing is on the process")
    message=output_string

    question="do you want treat null values?"
    return render_template("index.html", question=question, message=message, button_name="forth",list=False)


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

        if request.method == "POST":
            answerr = request.form.get("forth")
            if answerr == "yes":
                return redirect(url_for("drop_null_values"))
            elif answerr == "no":
                return redirect(url_for("uniquevalues"))


    output_info("These are the sum of null values in each columns of dataset")
    null_df = pd.DataFrame({'columns': (df.isnull().sum()).index, 'number of null values': (df.isnull().sum()).values})
    output_info(null_df.to_html(header="true", table_id="null_value_table"))

    
    message= output_string

    return render_template("index.html", question=question, message=message, button_name="fifth",suggestion=suggestion,list=False)

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
            output_info("{} columns has be droped from the dataset".format(selected_options))
            return redirect(url_for("drop_null_values"))

    question="Which of these columns would you like to drop?"
    message=output_string
    return render_template("index.html", question=question, message=message,list=True,options=options)

@app.route("/drop_null_values", methods=["GET", "POST"])
def drop_null_values():
    global df

    if request.method == "POST":
        answerr = request.form.get("sixth")
        if answerr == "yes":
            df = (df.dropna())
            output_info("In total {} rows has been dropped".format((str(df.isnull().values.sum()))))
            return redirect(url_for("duplicate"))
        elif answerr == "no":
            return redirect(url_for("uniquevalues"))

    message=output_string
    question="Do you want to remove null values"
    return render_template("index.html", question=question, message=message,list=False,button_name="sixth")

@app.route("/duplicate", methods=["GET", "POST"])
def duplicate():
    global df

    if request.method == "POST":
        answerr = request.form.get("seventh")
        if answerr == "yes":
            temp2=str("The dataset has {} duplicate rows".format((df.duplicated().sum())))
            output_info(temp2)
            return redirect(url_for("drop_duplicate_rows"))
        elif answerr == "no":
            return redirect(url_for("uniquevalues"))


    message=output_string
    question="Do you want to treat duplicate rows in the dataset?"
    return render_template("index.html", question=question, message=message,list=False,button_name="seventh")

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
        if request.method == "POST":
            answerr = request.form.get("eigth")
            if answerr == "yes":
                return redirect(url_for("drop_duplicate_rows_values"))
            elif answerr == "no":
                return redirect(url_for("uniquevalues"))

    message=output_string

    return render_template("index.html", question=question, message=message,list=False,button_name="eigth",suggestion=suggestion)

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
            output_info("{} columns has be droped from the dataset".format(selected_options))
            return redirect(url_for("drop_duplicate_rows_values"))

    question="Which of these columns would you like to drop?"

    message=output_string
    
    return render_template("index.html", question=question, message=message,list=True,options=options)

@app.route("/drop_duplicate_rows_values", methods=["GET", "POST"])
def drop_duplicate_rows_values():
    global df
    df=df.drop_duplicates()
    output_info("The duplicate rows has been removed")

    question="knslv"
    message=output_string
    return render_template("index.html", question=question, message=message,list=False)




if __name__ == "__main__":
    app.run(debug=True)