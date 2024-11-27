#!/bin/bash

# Установка sudo
apt install -y sudo

# Создание пользователя ubuntu и установка без пароля
useradd -m ubuntu
passwd -d ubuntu

# Добавление пользователя в sudoers с правами без пароля
echo "ubuntu ALL=(ALL) NOPASSWD: ALL" | tee -a /etc/sudoers > /dev/null

# Переключение на пользователя ubuntu и выполнение команд
su - ubuntu << 'EOF'

# Обновление пакетов и установка зависимостей
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip git

# Установка virtualenv
pip install virtualenv

# Добавление пути в переменную PATH
export PATH=$PATH:/home/ubuntu/.local/bin

# Создание виртуального окружения
virtualenv venv
source venv/bin/activate

# Клонирование репозитория и установка зависимостей
git clone https://github.com/vichesheiko/gptchain.git
cd gptchain
git checkout dev
pip install -r requirements-train.txt

# Напоминание создать .env файл перед запуском тренировки
echo "CREATE .env file before run train!!!"

EOF
