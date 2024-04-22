FROM python:3.8-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir pandas scikit-learn

EXPOSE 80

ENV NAME World

CMD ["python", "linear_reg.py", "df_smoothed.csv"]
