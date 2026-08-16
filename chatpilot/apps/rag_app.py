# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from typing import Optional
import json  # 添加这行到文件顶部的导入部分

from chatpilot.apps.auth_utils import get_current_user
from chatpilot.config import UPLOAD_DIR
from chatpilot.apps.web.models.users import Users

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化app状态
if not hasattr(app.state, 'user_files'):
    app.state.user_files = {}

@app.post("/doc")
def store_doc(
        collection_name: Optional[str] = Form(None),
        file: UploadFile = File(...),
        user=Depends(get_current_user),
):
    """接收上传文件并存储"""
    logger.debug(f"接收文件, 文件类型: {file.content_type}")
    try:
        # 检查文件大小
        file_size = 0
        contents = bytearray()
        while chunk := file.file.read(8192):
            contents.extend(chunk)
            file_size += len(chunk)
            # 检查是否超过3MB
            if file_size > 3 * 1024 * 1024:  # 3MB in bytes
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File is too large. Maximum allowed size is 3MB. "
                )
        
        # 获取文件名
        filename = file.filename
        
        # 创建用户专属目录
        user_upload_dir = f"{UPLOAD_DIR}/{user.id}"
        if not os.path.exists(user_upload_dir):
            os.makedirs(user_upload_dir)
            
        # 设置文件存储路径
        file_path = f"{user_upload_dir}/{filename}"
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(contents)
            
        # 更新用户的uploaded_files列表
        db_user = Users.get_user_by_id(user.id)
        if db_user:
            files = db_user.uploaded_files
            files.insert(0, filename)  # 将新文件添加到列表开头
            # 将 Python 列表转换为 JSON 字符串后再存储
            Users.update_user_by_id(user.id, {"uploaded_files": json.dumps(files)})
        
        return {
            "status": True,
            "filename": filename,
        }
        
    except HTTPException as he:
        logger.error(he)
        raise he
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
