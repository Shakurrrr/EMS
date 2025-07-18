from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy.exc import IntegrityError
from functools import wraps
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ========== Configuration ==========
API_TOKEN = os.getenv("API_TOKEN", "supersecret123")

# Prefer DATABASE_URL if provided (e.g. from Render)
DATABASE_URL = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")

if not DATABASE_URL:
    POSTGRES_USER = os.getenv("POSTGRES_USER", "flask_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "flask_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# ========== App & Extensions ==========
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

# ========== Token Decorator ==========
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-api-token')
        if not token or token != API_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# ========== Database Models ==========
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=True)
    category = db.Column(db.String(50), nullable=True)

class ContactSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Contact
        load_instance = True
        include_fk = True

contact_schema = ContactSchema()
contacts_schema = ContactSchema(many=True)

# ========== Initialize DB ==========
with app.app_context():
    db.create_all()

# ========== Routes ==========
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Contact Manager API!"}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

@app.route('/contacts', methods=['GET'])
@token_required
def get_contacts():
    contacts = Contact.query.all()
    return contacts_schema.jsonify(contacts), 200

@app.route('/contacts/search', methods=['GET'])
@token_required
def search_contacts():
    query = request.args.get('q')
    if not query:
        return contacts_schema.jsonify([]), 200
    results = Contact.query.filter(
        (Contact.name.ilike(f'%{query}%')) |
        (Contact.email.ilike(f'%{query}%'))
    ).all()
    return contacts_schema.jsonify(results), 200

@app.route('/contacts', methods=['POST'])
@token_required
def create_contact():
    data = request.get_json()
    new_contact = Contact(
        name=data.get('name', ''),
        email=data.get('email', ''),
        phone=data.get('phone', ''),
        category=data.get('category', '')
    )
    db.session.add(new_contact)
    try:
        db.session.commit()
        return contact_schema.jsonify(new_contact), 201
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "Email or phone already exists"}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/contacts/<int:id>', methods=['PUT'])
@token_required
def update_contact(id):
    contact = Contact.query.get_or_404(id)
    data = request.get_json()

    contact.name = data.get('name', contact.name)
    contact.email = data.get('email', contact.email)
    contact.phone = data.get('phone', contact.phone)
    contact.category = data.get('category', contact.category)

    try:
        db.session.commit()
        return contact_schema.jsonify(contact), 200
    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "Email or phone already exists"}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/contacts/<int:id>', methods=['DELETE'])
@token_required
def delete_contact(id):
    contact = Contact.query.get_or_404(id)
    db.session.delete(contact)
    db.session.commit()
    return jsonify({"message": "Contact deleted"}), 200

# ========== Entry Point ==========
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
