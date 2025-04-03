## Prerequisites

- Python 3.6 or higher installed
- Git installed
- (Optional) GitHub account for cloning private repositories

## Setup Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com/Aurel88-00/Encodec_Server](https://github.com/Aurel88-00/Encodec_Server)
cd Encodec_Server
```
* Make sure to clone the repo in the same directory as the frontend: [Encodec_Frontend](https://github.com/Aurel88-00/Encodec_Frontend)

## Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
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



