#!/bin/bash

env | egrep -v "^(PWD=|PATH=|OLDPWD=)" | sudo tee -a /etc/environment  # Ensure env variable are set
sudo service ssh start

exec "$@"
