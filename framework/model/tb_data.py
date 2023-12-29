

import mongoengine
from mongoengine import Document, StringField, IntField, DateTimeField, ListField, DictField

from config.config import CONFIG


def default_connect():
    db_config = CONFIG["mongodb_settings"]
    mongoengine.connect(db_config["database"], host=db_config["url"])
    
    

class Flow(Document):
    flowId = StringField(required=True)
    flowName = StringField(required=True)
    flowDescription = StringField()
    flowType = StringField()
    flowGraph= StringField()
    flowPrompt = StringField()
    flowInput = DictField()
    flowOutput = DictField()
    createdBy = StringField()
    createdAt = DateTimeField() 
    lastUpdatedBy = StringField()
    lastUpdatedAt = DateTimeField()
    accessLevel = IntField() # 0 for private; 1 for public
    tags = ListField() # [str0, str1, ...]
    webhook = DictField()




class Task(Document):
    
    taskId = StringField(required=True)
    flowId = StringField(required=True)
    status = IntField(required=True)   
    taskType = StringField()
    taskParams = DictField(required=True)
    createdBy = StringField()
    createdAt = DateTimeField()
    lastUpdatedAt = DateTimeField()
    webhook = DictField()
    
    
    
    
class TaskReuslt(Document):
    
    taskId = StringField(required=True)
    status = IntField(required=True)
    startTime = DateTimeField()
    endTime = DateTimeField()
    result = DictField()
    error = StringField()
    
