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
    
    # return {'text_1':text_1 , 'text_2': text_2}
    return templates.TemplateResponse('index.html', {'request': request, 'username': username, 'password': password})

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

****
```Python
```

---

****
```Python
```

---
