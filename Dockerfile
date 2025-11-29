FROM tensorflow/tensorflow:2.16.2

WORKDIR /app

COPY interface/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

COPY . .

WORKDIR /app/interface

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
