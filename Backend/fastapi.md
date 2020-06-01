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
