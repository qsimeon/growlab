FROM python:3.11-slim
WORKDIR /app
COPY deploy/server.py /app/server.py
COPY web/ /app/web/
ENV PORT=8080
EXPOSE 8080
CMD ["python", "-u", "/app/server.py"]
