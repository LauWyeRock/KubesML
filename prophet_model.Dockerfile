
FROM python:3.8-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir pandas prophet

EXPOSE 80

ENV NAME World

CMD ["python", "prophet_model.py", "df_smoothed.csv"]
