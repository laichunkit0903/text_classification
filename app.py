#from sklearn.externals import joblib
import joblib
from flask import Flask, jsonify,request,json
from flask import render_template
import spacy
import en_core_web_sm
import pandas as pd
import sys
sys.modules['sklearn.externals.joblib'] = joblib

nlp = spacy.load("en_core_web_sm")
svm_clf = joblib.load('NB_textclassification_new.sav')
#lotame_clf = joblib.load('GA_NST_NB_lotame.sav')

def entity_tag(article):
    doc = nlp(article)
    label = []
    text = []
    full=[]
    etity_label = []
    entity_list = ['PERSON','ORG','GPE','PRODUCT']
    for ent, i in zip(doc.ents,range(0,len(doc))):
        label.append(ent.label_)
        text.append(ent.text)
        if label[i] in entity_list:
            full.append(text[i])
            etity_label.append(label[i])
    
    df = pd.DataFrame({'Entity Label':etity_label ,'Entity':full})
    df = df.drop_duplicates(keep='first')
    #entity = list(set(full))
    #etity_label_unique =  list(set(etity_label))
    return df




app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/", methods =['GET','POST'])
def api_predict():
     json = request.form['text']
     strs = [""]
     strs.append(json)
     strs.pop(0)
     #query_df = pd.DataFrame(json,index=[0])
     ##query = pd.get_dummies(query_df)
     entity_df = entity_tag(json)
     #entity = ' , '.join(entity)
     #entity_df = pd.DataFrame({'Entity Label':etity_label_unique ,'Entity':entity})
     prediction = svm_clf.predict(strs)
     prediction = ''.join(prediction)
     #lotame_prediction = lotame_clf.predict(strs)
     #lotame_prediction = ''.join(lotame_prediction)
     #prediction = prediction.tolist()
     #predicted_result = {'prediction': list(prediction)}
     return render_template('results.html',results = prediction ,df = entity_df.to_html(classes='demographic'))
 
    
if __name__ == '__main__':
    app.run()
