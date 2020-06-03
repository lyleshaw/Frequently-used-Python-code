# fastapi

**启动服务**
```Python
from fastapi import FastAPI

app = FastAPI()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=80)
```
---

**jinja2渲染**
```Python
from starlette.requests import Request
from fastapi import FastAPI
from starlette.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'hello': 'HI...'})
```

```html
<body>
    <h1>HELLO FastAPI...</h1>
    <h1>{{ hello }}</h1>
    <h1>{{ item_id }}</h1>
</body>
```

---

**“./u=1&n=2”式提交表单**
```Python
from starlette.requests import Request
from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/user/")
async def form_text(request: Request, username: str = Form(...), password: str = Form(...)):
    
    print('username',username)
    print('password',password)
    
    return templates.TemplateResponse('index.html', {'request': request, 'username': username, 'password': password})

```

---

**文件传输**
```Python
from typing import List
from starlette.requests import Request
from fastapi import FastAPI, Form, File, UploadFile
from starlette.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.post("/files/")
async def files(
                    request: Request,
                    files_list: List[bytes]         = File(...),
                    files_name: List[UploadFile]    = File(...),
                ):
    return templates.TemplateResponse("index.html", 
            {
                "request":      request,
                "file_sizes":   [len(file) for file in files_list], 
                "filenames":    [file.filename for file in files_name],    
             })
```

---

**参数验证**
```Python
app = FastAPI()
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

@app.get("/i/")
async def i(A: str = 'HI..', B: str = 'Hello..', C: str = 'He..'):
    return {'cc': A+B+C},{'dd': B+C}
    
@app.get("/ii/")
async def ii(A: int = 0, B: int = 10, C: int = 20):
    return {'cc': A+B+C},{'dd': B+C}

@app.get("/iii/")
async def iii(A: int = 0, B: int = 10, C: int = 20):
    return 'A+B+C',A+B+C
   
# bool与类型转换
@app.get("/xxx/{item_id}")
async def xxx(item_id: str, QQ: str = None, SS: bool = False):
    item = {"item_id": item_id}
    if QQ:
        item.update({"QQ": QQ})
    if not SS:  # 如果SS是假
        item.update(
            {"item_id": "This is SSSSSSS"}
        )
    return item

#多路径 和 查询参数 和 必填字段
@app.get("/user/{user_id}/item/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str = None, short: bool = False
):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item
```

---10

****
```Python
```

---

****
```Python
```

---

****
```Python
```

---

****
```Python
```

---

****
```Python
```

---

****
```Python
```

---

****
```Python
```

---

**json转换**
```Python
from fastapi.encoders import jsonable_encoder
class Item(BaseModel):
    title: str
    timestamp: datetime
    description: str = None
json_compatible_item_data = jsonable_encoder(item)
```

---

****
```Python
```

---

**CORS**
```Python
from starlette.middleware.cors import CORSMiddleware
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8000"
    
]

app.add_middleware(         # 添加中间件
    CORSMiddleware,         # CORS中间件类
    allow_origins=origins,  # 允许起源
    allow_credentials=True, # 允许凭据
    allow_methods=["*"],    # 允许方法
    allow_headers=["*"],    # 允许头部
)
```

---

**数据库配置**
```Python
from typing import List
from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,Session
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

SQLALCHEMY_DATABASE_URL = "sqlite:///db_test_3.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}) #仅sqlite需要

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) #创建数据库会话
Base = declarative_base() #创建orm基类

class M_User(Base):  #继续base创建数据库
    __tablename__ = "users" #表名

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String) 
    is_active = Column(Boolean, default=True)

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    class Config:
        orm_mode = True

Base.metadata.create_all(bind=engine) #创建数据库
```

---

**数据库依赖**
```Python
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
        print('数据库已关闭')
```

---

**数据库-增**
```Python
```

---

**数据库-删**
```Python
```

---

**数据库-查**
```Python
```

---

**数据库-改**
```Python
```

---

****
```Python
```

---

****
```Python
```

---

**数据库操作**
```python
from typing import List
from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,Session
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
# SELECT * FROM users 

"""↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓    数据库操作初始化    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"""
SQLALCHEMY_DATABASE_URL = "sqlite:///db_test_3.db"
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# 仅适用于SQLite。其他数据库不需要。 链接参数：检查同一条线？ 即需要可多线程

# 通过sessionmaker得到一个类，一个能产生session的工厂。
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) 
             # 会话生成器   自动提交            自动刷新
Base = declarative_base() # 数据表的结构用 ORM 的语言描述出来

class M_User(Base):  #声明数据库某表的属性与结构
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True) # 主键
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String) 
    is_active = Column(Boolean, default=True)

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    class Config:
        orm_mode = True

Base.metadata.create_all(bind=engine)

app = FastAPI()

"""↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓    数据库操作（依赖项）    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"""
def get_db():
    try:
        db = SessionLocal() # 这时，才真正产生一个'会话'，并且用完要关闭
        yield db            # 生成器
    finally:
        db.close()
        print('数据库已关闭')


"""↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓    数据库操作方法    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"""
# 通过id查询用户信息
def get_user(db: Session, user_id: int):
    CCCCCC = db.query(M_User).filter(M_User.id == user_id).first()
    print(CCCCCC)           # 过滤器
    return CCCCCC

# 新建用户（数据库）
def db_create_user(db: Session, user: UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = M_User(email=user.email, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()     # 提交即保存到数据库
    db.refresh(db_user) # 刷新
    return db_user

"""↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓    post和get请求    ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"""
# 新建用户(post请求)
@app.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Depends(get_db)使用依赖关系可防止不同请求意外共享同一链接
    return db_create_user(db=db, user=user)

# 用ID方式读取用户
@app.get("/users/{user_id}", response_model=User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    print(db_user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return db_user


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```