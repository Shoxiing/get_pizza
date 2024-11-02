FROM python:3.11

WORKDIR /tg_bot

COPY . .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

RUN apt-get update && apt-get install -y ffmpeg


CMD ["python", "tg_bot_app.py"]
