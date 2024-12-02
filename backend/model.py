# models.py

from extension import db  # Import db from extensions.py

class ImageHash(db.Model):
    __tablename__ = 'image_hashes'
    id = db.Column(db.Integer, primary_key=True)
    hash = db.Column(db.String, unique=True, nullable=False)
    predicted_class = db.Column(db.Integer, nullable=False)

