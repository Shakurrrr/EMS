# 📝 Contact Manager API

A simple Flask-based RESTful API for managing contacts using **PostgreSQL**, **SQLAlchemy**, and **Docker**. Includes search, CRUD operations, validation, and a Marshmallow schema.

---

## 📦 Features

- Add, update, delete, and list contacts
- Search contacts by name or email
- Marshmallow for serialization
- Docker + PostgreSQL integration
- Retry logic for PostgreSQL startup

---

## 🚀 Quickstart (Local via Docker)

### 1. Clone this Repo

```bash
git clone https://github.com/yourusername/contact-manager-api.git
cd contact-manager-api
```

### 2. Create `.env` File

```env
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_DB=mydb
POSTGRES_HOST=db
```

### 3. Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app.py
└── .env
```

---

## 🐳 Docker Setup

### Build & Run Containers

```bash
docker-compose down -v
docker-compose up --build
```

### Access the API

Visit: [http://localhost:5000](http://localhost:5000)

---

## 📂 API Endpoints

| Method | Endpoint                 | Description                |
|--------|--------------------------|----------------------------|
| GET    | `/`                      | Welcome message            |
| GET    | `/health`                | Health check               |
| GET    | `/contacts`              | Get all contacts           |
| GET    | `/contacts/search?q=...` | Search by name or email    |
| POST   | `/contacts`              | Create a new contact       |
| PUT    | `/contacts/<id>`         | Update a contact           |
| DELETE | `/contacts/<id>`         | Delete a contact           |

---

## 🧪 Example POST Request (Using Postman)

**URL**: `http://localhost:5000/contacts`

**Body** (JSON):
```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "phone": "1234567890"
}
```

---

## 🛠️ Code Highlights

### 📁 app.py

```python
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os, time, psycopg2
from sqlalchemy.exc import IntegrityError

app = Flask(__name__)
# DB Config & Retry Logic...
# Contact Model & Marshmallow Schema...
# CRUD Routes...
```

### 📄 Dockerfile

```Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "app.py"]
```

### 📄 docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - .:/app
    environment:
      - FLASK_ENV=development
    env_file:
      - .env
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## 📚 Tech Stack

- **Python** + **Flask**
- **SQLAlchemy** + **PostgreSQL**
- **Docker Compose**
- **Marshmallow** for serialization

---

## 🧑‍💻 Author

Built by Alameen Yahya Harande • June 27, 2025

---

## 📄 License

MIT License — free to use and modify.
