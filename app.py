from flask import Flask,redirect,url_for,render_template,request
import pandas
import htmltest
from htmltest import main
app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/forms')
def forms():
    return render_template('forms.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/success/<int:score>')
def success(score):
    res=""
    if score>=27:
        level=3
    elif score>=14 and score<=26:
        level=2
    else:
        level=1
    return_list = htmltest.main(score,level)
    print(return_list)
    rec_1 = return_list[0]
    rec_2 = return_list[1]
    rec_3 = return_list[2]
    res = str(', '.join(return_list))
    return render_template('result.html', rec_1 = rec_1, rec_2 = rec_2, rec_3 = rec_3)

@app.route('/submit',methods=['POST','GET'])
def submit():
    total_score=0
    if request.method=='POST':
         a=float(request.form['val0'])
         b=float(request.form['val1'])
         c=float(request.form['val2'])
         d=float(request.form['val3'])
         e=float(request.form['val4'])
         f=float(request.form['val5'])
         g=float(request.form['val6'])
         h=float(request.form['val7'])
         i=float(request.form['val8'])
         j=float(request.form['val9'])

         total_score=(a+b+c+d+e+f+16-(g+h+i+j))
    res=""
    
    return redirect(url_for('success',score=total_score))
         
if __name__=='__main__':
    app.run(debug=True)
