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
import time

app = Flask(__name__)

df = pd.read_csv("application_record.csv")

def ouput_string(inp):
    output_info=output_info+"<br>"+inp
    return output_info
    

    

@app.route("/", methods=["GET", "POST"])
def home():
    global perform_exploration

    question = "Do you want to perform data exploration?"
    message = ""
    perform_exploration = None

    if request.method == "POST":
        answer = request.form.get("answer")
        if answer == "yes":
            perform_exploration = True
            return redirect(url_for("explore"))
        elif answer == "no":
            perform_exploration = False
            message = "Proceeding to next step"
            question = "Do you want to perform preprocessing?"
            return render_template("index.html", question=question, message=message)

    return render_template("index.html", question=question, message=message)

@app.route("/explore")
def explore():
    global perform_exploration

    message = ""
    question = "Do you want to perform data exploration?"

    if perform_exploration == True:
        output = "The shape of the given dataset {}".format(str(df.shape))
        output = output + "<br>The dataset has {} rows and {} columns".format(df.shape[0], df.shape[1])
        message = output
        question = "Want to analyse the unique values in each categorical column?"
    elif perform_exploration == False:
        message = "Proceeding to next step"
        question = "Do you want to perform preprocessing?"

    return render_template("index.html", question=question, message=message)

if __name__ == "__main__":
    app.run(debug=True)


