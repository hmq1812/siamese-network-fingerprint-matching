from flask import Flask
from flask_restful import Resource, reqparse, Api
import socket
import werkzeug
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

from matching import FingerPrintMatching

app = Flask(__name__)
api = Api(app)

ALLOWED_PROFILE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = FingerPrintMatching()

def allowed_file(filename):
    file_type = filename.split(".")[1]
    if file_type in ALLOWED_PROFILE_EXTENSIONS:
        return True
    return False


class FingerPrintMatchingAPI(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('file1', type=werkzeug.datastructures.FileStorage, location='files', help="No 'file' header found")
    parser.add_argument('file2', type=werkzeug.datastructures.FileStorage, location='files', help="No 'file' header found")
    parser.add_argument('url1', type=str)
    parser.add_argument('url1', type=str)

    def post(self):
        data = self.parser.parse_args()
        file1 = data['file1']
        file2 = data['file2']

        if file1 is None or file1.filename == '' or file2 is None or file2.filename == '':
            return {"errorCode": 3, "errorMessage": "Incorrect Image Format"}, 400

        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return {"errorCode": 3, "errorMessage": "Incorrect Image Format"}, 400

        try:
            print('tryingg yhihihii')
            byte_image1 = file1.read()
            print(type(byte_image1))
            byte_image2 = file2.read()
            image1 = Image.open(BytesIO(byte_image1))
            image2 = Image.open(BytesIO(byte_image2))
            print(type(image1))
            embedding1 = model.get_embedding(image1)
            print(embedding1)
            embedding2 = model.get_embedding(image2)

            matching = model.match(embedding1, embedding2)
            
            print(matching)

        except Exception as e:
            print(e)
            return {"errorCode": 1, "errorMessage": "Unreadable"}, 400
        return {"errorCode": 0, "errorMessage": "Success", "matching": matching}, 200
                
    def get(self):
        data = self.parser.parse_args()
        file1 = data['url1']
        file2 = data['url2']

        if file1 is None or file1 == '' or file2 is None or file2 == '':
            return {"errorCode": 2, "errorMessage": "Url is unavailable"}, 400

        try:
            byte_image1 = urlopen(file1).read()
            byte_image2 = urlopen(file2).read()
        except:
            return {"errorCode": 2, "errorMessage": "Url is unavailable"}, 400

        try:
            image = Image.open(BytesIO(byte_image1))  
            image = Image.open(BytesIO(byte_image2))  
            embedding1 = model.get_embedding(image1)
            embedding2 = model.get_embedding(image2)
            matching = model.match(embedding1, embedding2)
            
        except:
            return {"errorCode": 1, "errorMessage": "Unreadable"}, 400
        return {"errorCode": 0, "errorMessage": "Success", "matching": matching}, 200     


api.add_resource(FingerPrintMatchingAPI, "/fingerprintMatching")

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


if __name__ == '__main__':
    app.run(host=get_ip(), port=5000, debug=True)
