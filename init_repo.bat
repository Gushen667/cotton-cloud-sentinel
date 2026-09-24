@echo off
set "GIT="C:\Program Files\Git\cmd\git.exe""
set "REPO=D:\AI测试\whitefly"

echo === 1 config ===
%GIT% config --global user.name "cottoncloud"
%GIT% config --global user.email "mianyunshao@example.com"

echo === 2 init ===
%GIT% -C "%REPO%" init -b main

echo === 3 add ===
%GIT% -C "%REPO%" add -A

echo === 4 commit ===
%GIT% -C "%REPO%" commit -m "MianYunShao streamlit deploy"

echo === 5 list files ===
%GIT% -C "%REPO%" ls-files

echo === DONE ===
