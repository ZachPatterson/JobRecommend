FROM python:3.9

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8501

COPY . .

COPY load_env.sh /app/load_env.sh
RUN chmod +x /app/load_env.sh

ENTRYPOINT ["/app/load_env.sh"]
CMD ["streamlit", "run", "app.py"]
