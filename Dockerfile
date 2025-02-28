FROM ultralytics/yolov5:latest

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "srcs/Blackjack.py"]
