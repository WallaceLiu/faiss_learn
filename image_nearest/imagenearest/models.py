from sqlalchemy import (
    Column,
    Index,
    Integer,
    Text,
    orm,
    MetaData,
    Table,
    DateTime,
    LargeBinary,
    ForeignKey,
    Table,
    Boolean,
    func,
    String,
    BigInteger,
    Numeric,
)
from sqlalchemy.orm import relationship
from .meta import Base


import datetime 

class EssentialColMixin(object):
    id = Column(Integer, primary_key=True)
    created = Column(DateTime, default=datetime.datetime.utcnow, server_default="now")

class Product(Base, EssentialColMixin):
    __tablename__ = 'products'
    name = Column(Text)
    _description = Column(Text)
    
    