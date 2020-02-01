import numpy as np
from flask import Flask, request, jsonify, render_template,url_for,abort,flash,redirect
from werkzeug import secure_filename
import pickle
import pandas as pd
import os


app = Flask(__name__)
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500