# app/Dockerfile

FROM python:3.7

WORKDIR C:/Users/User/Desktop/app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY  app.py ./
COPY  saved_CNN_model.h5 ./
COPY  weights.npy ./

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]