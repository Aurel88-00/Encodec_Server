## Prerequisites

- Python 3.6 or higher
- Git 

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Aurel88-00/Encodec_Server.git
cd Encodec_Server
```
* Make sure to clone the repo in the same directory as the frontend: [Encodec_Frontend](https://github.com/Aurel88-00/Encodec_Frontend)

## Create and Activate Virtual Environment
```bash
python -m venv venv
venv\bin\activate
```
## Install Dependencies
```bash
pip install -r requirements.txt
```
If you don't have a requirements.txt file, create one using:

```bash
pip freeze > requirements.txt
```
## Start the application
```bash
uvicorn main:app --reload
```



