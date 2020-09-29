from flask import Flask,request,jsonify
from flask_cors import CORS
import recommendations 
from recommendations import top
app = Flask(__name__)
CORS(app) 

top5=top()
topytop= str(top5)

        
@app.route('/authors', methods=['GET'])

def authors():
    results = recommendations.SimilarAuthor(request.args.get('records'))
    return jsonify(results)

@app.route('/books', methods=['GET'])

def books():
    results = recommendations.SimilarBooks(request.args.get('records'))
    return jsonify(results)

    
@app.route('/top5', methods=['GET'])

def topfive():
    return jsonify(topytop)

if __name__=='__main__':
    app.run()