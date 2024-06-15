cd /mnt/workspace

git clone https://www.aiops.cn/gitlab/aiops-challenge/aiops-2024-submit.git

git clone -b glm https://github.com/issaccv/aiops24-RAG-demo.git

cd aiops24-RAG-demo

bash run.sh

cd demo

pip install -r requirements.txt

python main.py

