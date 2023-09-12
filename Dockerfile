FROM python:3.9.6

WORKDIR /app

COPY requirements.txt /app/

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

ENV FLASK_ENV=production

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN python3 update.py -v

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
