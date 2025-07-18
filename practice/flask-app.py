from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from functools import wraps
from dotenv import load_dotenv
import os

#  Load environment variables from .env file
load_dotenv()

#  Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

#  Set up extensions
db = SQLAlchemy(app)
ma = Marshmallow(app)

#  User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)

#  Marshmallow schema
class UserSchema(ma.SQLAlchemySchema):
    class Meta:
        model = User

    id = ma.auto_field()
    username = ma.auto_field()

# Schema instances
user_schema = UserSchema()
users_schema = UserSchema(many=True)

#  Create DB tables
with app.app_context():
    db.create_all()

#  Custom token decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')

        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        if token != 'mysecrettoken':
            return jsonify({'error': 'Invalid token!'}), 403

        return f(*args, **kwargs)
    return decorated

#  Home route
@app.route('/')
def index():
    return 'API is running!'

#  Add user route
@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    username = data.get('username')

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'error': 'Username already exists'}), 409

    new_user = User(username=username)
    db.session.add(new_user)
    db.session.commit()

    return user_schema.jsonify(new_user), 201

#  Get all users route
@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return users_schema.jsonify(users), 200

#  Protected route with token
@app.route('/secret', methods=['GET'])
@token_required
def secret_data():
    return jsonify({'message': 'You accessed protected data!'}), 200

#  Run the app
if __name__ == '__main__':
    app.run(debug=True)
