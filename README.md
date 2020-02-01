# clusteringApplicationUsingFlask
Clustering based machine learning application using Flask

This is an unsupervised learning based Clustering web tool that has been used to cluster microstructures in Steel based on their physical properties. Files can be uploaded in .xlsx or .csv format. The tool supports five different clustering algorithms in combination with dimensionality reduction. The dimensionality reduction algorithm used here is Principal Component Analysis (PCA).

This web tool is developed as part of the coursework - Software Engineering, for the Chair for Functional Materials, Saarland University.

More information about the department can be found [here](https://www.fuwe.uni-saarland.de/).


## Setting up the environment

This project is built using Python 3.8. 

1. Clone the directory
```
git clone https://github.com/microclustering
```

2. Install dependencies by running the following command:
```
pip install -r requirements.txt
```

## Launching the Webtool

### Running from Command Line
```
cd microclustering_final version/microclustering_WM
python app.py
```
### Creating and running a Windows Batch File
1. Navigate to the runscript batch file.
2. Replace the placeholders {"path-to-python.exe" "path-to-app.py"} with the actual path and save the .bat file
3. Run the saved runscript.bat file for launching the application



## Navigating through the Webtool
 
1. The application opens in default browser on the Home page.
2. Select the Upload tab and upload the dataset file in .xlsx or .csv format. After uploading, Summary of the uploaded dataset can be checked by clicking Get summary button.
3. When the file is uploaded, navigate to Cluster tab and select from the five listed lustering algorithms. The maximum number of clusters that can be chosen is 50. Check the box to use dimensionality reduction with PCA algorithm for the uploaded dataset. 
4. A success hyperlink can be seen on the screen and clicking it will navigate you to the download page.
5. Summary of clustering results can be seen and can be downloaded using download summary button. Also, the results can be saved in .csv or .xlsx format locally.


## License
This code is released under the terms of the [MIT license](https://github.com/microclustering/project25/blob/master/LICENSE).
