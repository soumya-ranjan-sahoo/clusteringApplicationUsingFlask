import numpy as np
from flask import Flask, request, jsonify, render_template,url_for,abort,flash,redirect,send_file, send_from_directory
from werkzeug import secure_filename
import pickle
import pandas as pd
import os
from all import*
import xlrd
import csv
import webbrowser
from threading import Timer
#from waitress import serve

# variables
file_name = ""
t_algo=None
Upload = None
file_name = None
result = pd.DataFrame()
home_path = os.getcwd() + '/'
app = Flask(__name__)
t_k_clusters=None


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    
    return render_template('500.html'), 500
	
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response
def csv_from_excel(excelFile):
    wb = xlrd.open_workbook(excelFile)
    #sh = wb.sheet_by_name('Sheet1')
    sh = wb.sheet_by_index(0)
    your_csv_file = open(home_path + 'uploadedFile.csv', 'w', encoding = 'utf-8')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    print("converting to .csv")
    your_csv_file.close()

# runs the csv_from_excel function:

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')
    #ie=webbrowser.get('C:\\Program Files\\Internet Explorer\\iexplore.exe')
    #ie.open('http://127.0.0.1:5000/')
	  
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload')
def upload():
   global Upload, file_name
   if Upload:
      return render_template('upload1.html', success_text=' Already uploaded {}'.format(file_name))
   else:
      return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   global Upload
   global file_name
   if request.method == 'POST':
      if os.path.isfile('uploadedFile.csv'):
        print("removing already loaded file ...")
        os.remove('uploadedFile.csv');Upload = True
      f = request.files['file']
      file_name = secure_filename(f.filename)
      print(file_name)
      f.save(file_name)
      if ".csv" in file_name:
        os.rename(file_name,'uploadedFile.csv')
      if ".xlsx" in file_name:
        csv_from_excel(file_name)
      print("converted to csv ...");return render_template('upload.html', success_text=' Successfully uploaded {}'.format(file_name))
   else:
	    Upload = False;return render_template('upload.html', success_text=' Not uploaded {}'.format(file_name))
    
   

@app.route('/summary', methods = ['GET', 'POST'])
def get_summary():
    global Upload
    if request.method == 'GET' and Upload:
        file = r'uploadedFile.csv'
        #file = r'Database-Objects.csv'   #r'+file_name # hard-coded, need to be generalized for uploaded file
        try:
            df = pd.read_csv(file,encoding = "ISO-8859-1")
            print("uploading summary ...")
            df_head = df.head()
            print("returning template")
            return render_template('upload.html',  tables=[df_head.to_html(classes='data')], titles=df.columns.values)
        except:
            print("except")
            abort(404)
            
@app.route('/cluster')
def cluster():
    return render_template('cluster.html')

@app.route('/cluster_algo', methods=['GET', 'POST'])
def cluster_algo():
    algo = request.form.get('algo_select')
    return render_template('cluster.html',algo_text='You selected {}'.format(algo))


@app.route('/cluster_features', methods=['GET', 'POST'])
def cluster_features():

    if request.form.get("DR"):
        pca_checked = True
    else:
        pca_checked = "use_all"


    return render_template('cluster.html',k_features ='You opted for {} features for clustering'.format(DR))

@app.route('/cluster_k', methods=['GET', 'POST'])
def cluster_k():
    global Upload
    if request.form.get("DR"):
        pca = True
    else:
        pca = False
    algo = request.form.get('algo_select')
    global t_algo
    t_algo=algo
    k_clusters = request.form.get('k_select')
    global t_k_clusters
    t_k_clusters=k_clusters
    if Upload:
       cluster_obj,result =  decision(algo,k_clusters,pca) ####Need to make this varibale global o it is accessible to download
    #download(clus,result)
    download_data(cluster_obj,result,k_clusters,'.xlsx')
    return render_template('cluster.html',k_text='Your file has been processed using {} and is ready for downloading'.format(algo),k_loading = "Loading..")
    #return render_template('cluster1.html',k_text='Your file has been processed using {} and is ready for downloading'.format(algo),k_loading = "Loading..")

@app.route('/terminate_clustering')
def terminate_clustering():
    return render_template('cluster.html')

@app.route('/download')
def download():
    #download_data(clus, result,'.xlsx')
    #print("generating summary:")
    #gen_summary()
    return render_template('download.html')
    
@app.route('/output_summary')
def output_summary():
    #download_data(clus, result,'.xlsx')
    #print("generating summary:")
    #gen_summary()
    global t_algo, t_k_clusters
    #print(t_algo,t_k_clusters);print("Hello")
    path = home_path+'summary/'+t_algo+str(t_k_clusters)+'.txt'
    summary = open(path, "r")
    content = summary.read()
    return render_template('download.html',output_summary = content)

@app.route('/download_summary')
def download_summary():
    global t_algo, t_k_clusters
    print("Not Working")
    path = home_path+'summary/'+t_algo+str(t_k_clusters)+'.txt'
    print(home_path+path)
	#filename=home_path+path
    return send_file(path, as_attachment=True)
    #return render_template('download.html',download_text='Successfully downloaded')

@app.route('/download_output')
def download_output():
    global t_algo, t_k_clusters
    path = home_path+"/output/output_"+t_algo+"_"+ t_k_clusters+ ".xlsx"
    return send_file(path, as_attachment=True)
    #return render_template('download.html',download_text='Successfully downloaded')


if __name__ == "__main__":
	Timer(1, open_browser).start();
	app.run(debug=False)
	#serve(app, host='0.0.0.0', port=8000)
