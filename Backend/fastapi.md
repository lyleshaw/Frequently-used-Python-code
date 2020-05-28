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
