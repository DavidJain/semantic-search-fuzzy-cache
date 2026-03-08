import os
import re
from tqdm import tqdm
import logging

# Set up simple logging to see skip reasons
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsGroupDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.documents = []
        self.labels = []

    def clean_text(self, text):
        # remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)
        # remove URLs
        text = re.sub(r"http\S+|www\S+", " ", text)
        # remove quoted reply lines
        text = re.sub(r"^>.*$", " ", text, flags=re.MULTILINE)
        # remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def load_dataset(self):
        # Clear existing data if re-loading
        self.documents = []
        self.labels = []
        
        if not os.path.exists(self.dataset_path):
            logger.error(f"Path not found: {self.dataset_path}")
            return []

        categories = [d for d in os.listdir(self.dataset_path) 
                      if os.path.isdir(os.path.join(self.dataset_path, d))]

        for category in tqdm(categories, desc="Loading Categories"):
            category_path = os.path.join(self.dataset_path, category)
            files = os.listdir(category_path)

            for file in files:
                file_path = os.path.join(category_path, file)
                try:
                    # Using latin1 is fine for 20newsgroups, but we handle errors just in case
                    with open(file_path, "r", encoding="latin1") as f:
                        raw_text = f.read()
                        cleaned_text = self.clean_text(raw_text)

                        # Only keep meaningful documents
                        if len(cleaned_text) > 50:
                            self.documents.append(cleaned_text)
                            self.labels.append(category)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path} due to error: {e}")
                    continue

        return self.documents

    def get_documents(self):
        return self.documents

    def get_labels(self):
        return self.labels