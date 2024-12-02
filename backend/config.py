import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # General Config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'
    DEBUG = True

    # Database Config
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'image_hashes.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload Config
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    UPLOAD_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif']
    UPLOAD_PATH = os.path.join(basedir, 'uploaded_images')
