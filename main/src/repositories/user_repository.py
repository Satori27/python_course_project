import os
from typing import Type

from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, Session

DB_USER = os.getenv("POSTGRES_USER", "myuser")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mypassword")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "mydatabase")


DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

from .models import BaseDeclaration

# ORM Models
class User(BaseDeclaration):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    disabled = Column(Boolean, default=False)
    roles = relationship("Role", secondary="user_roles", back_populates="users")

class Role(BaseDeclaration):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    users = relationship("User", secondary="user_roles", back_populates="roles")

class UserRole(BaseDeclaration):
    __tablename__ = "user_roles"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), primary_key=True)

# Database utility functions
def get_user_by_username(db, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_id(db, user_id: int):
    try:
        return db.query(User).filter(User.id == user_id).first()
    except Exception as e:
        print(f"Error: user_repository.get_user_by_id {e}")
def get_user_roles(db, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    return [role.name for role in user.roles] if user else []

def create_user(db, username: str, password_hash: str, full_name: str = None) -> User:
    new_user = User(username=username, full_name=full_name, hashed_password=password_hash)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def assign_role_to_user(db, user_id: int, role_name: str):
    role = db.query(Role).filter(Role.name == role_name).first()
    if not role:
        role = Role(name=role_name)
        db.add(role)
        db.commit()
        db.refresh(role)

    user = db.query(User).filter(User.id == user_id).first()
    if user and role not in user.roles:
        user.roles.append(role)
        db.commit()

def assign_roles_to_user(db, user_id: int, roles: list[str]):
    if (roles == None or user_id == None):
        return
    for role in roles:
        assign_role_to_user(db, user_id, role)

def get_all_users(db: Session, limit: int) -> list[Type[User]]:
    return db.query(User).limit(limit).all()

# Dependency for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
