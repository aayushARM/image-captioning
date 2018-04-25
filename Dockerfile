FROM gcr.io/tensorflow/tensorflow:1.6.0-py3

RUN pip uninstall -y jupyter 
ADD . ./
RUN pip install -r requirements.txt
CMD gunicorn -b :$PORT flask_server:app
