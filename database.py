from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv
# Automatically load the .env file from the current directory or parent directories
load_dotenv()
DATABASE_NAME= os.getenv('DATABASE_NAME')
DATABASE_PASSWORD=os.getenv('DATABASE_PASSWORD')
DATABASE_PORT=os.getenv('DATABASE_PORT')
DATABASE_HOST=os.getenv('DATABASE_HOST')
DATABASE_USER=os.getenv('DATABASE_USER')
DATABASE_URL = f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
engine= create_engine(DATABASE_URL)
session_local= sessionmaker(autoflush=False, autocommit=False, bind=engine)
Base= declarative_base()
