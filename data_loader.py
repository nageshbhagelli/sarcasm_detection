import pandas as pd
import json
import os
import requests
from tqdm import tqdm
from newspaper import Article
import re

class SarcasmDataLoader:
    def __init__(self, file_path, subset_size=None):
        self.file_path = file_path
        self.subset_size = subset_size
        self.df = None

    def load_data(self):
        """Loads the JSON dataset and performs basic cleaning."""
        print(f"Loading data from {self.file_path}...")
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        self.df = pd.DataFrame(data)
        
        if self.subset_size:
            print(f"Sampling {self.subset_size} records for speed.")
            self.df = self.df.sample(n=self.subset_size, random_state=42).reset_index(drop=True)
            
        print(f"Dataset Loaded. Total records: {len(self.df)}")
        self._check_distribution()
        return self.df

    def _check_distribution(self):
        """Prints class distribution."""
        dist = self.df['is_sarcastic'].value_counts(normalize=True) * 100
        print("\n--- Class Distribution ---")
        print(f"Sarcastic: {dist.get(1, 0):.2f}%")
        print(f"Not Sarcastic: {dist.get(0, 0):.2f}%")
        print("--------------------------\n")

    def add_context(self, scrape_limit=100):
        """
        Adds a 'context' column. 
        Tries to scrape first (up to scrape_limit), then falls back to synthetic context.
        """
        print(f"Adding context (Scrape limit: {scrape_limit} samples, others will use fallback)...")
        contexts = []
        
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            headline = row['headline']
            url = row['article_link']
            
            context = ""
            # Try scraping for a limited number of samples to save time/resources
            if i < scrape_limit:
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    # Use the first 200 characters of the text as summary/context
                    context = article.text[:300].replace('\n', ' ')
                    if len(context) < 10: # If scraping returned garbage/nothing
                        context = self._generate_synthetic_context(headline, url)
                except Exception:
                    context = self._generate_synthetic_context(headline, url)
            else:
                # Fallback for the rest of the dataset
                context = self._generate_synthetic_context(headline, url)
            
            contexts.append(context)
            
        self.df['context'] = contexts
        return self.df

    def _generate_synthetic_context(self, headline, url):
        """Generates a plausible context based on headline and source domain."""
        # Detect domain
        domain = "General News"
        if "theonion.com" in url:
            domain = "Satire and Comedy"
        elif "huffingtonpost.com" in url:
            domain = "Standard News Report"
            
        # Extract keywords (simple regex for capitalized words or longest words)
        words = re.findall(r'\w+', headline)
        keywords = sorted(words, key=len, reverse=True)[:3]
        keyword_str = ", ".join(keywords)
        
        return f"Source: {domain}. This article discusses topics related to {keyword_str}. " \
               f"The narrative frame focuses on the following: {headline}."

if __name__ == "__main__":
    # Test loading
    loader = SarcasmDataLoader('Sarcasm_Headlines_Dataset.json', subset_size=50)
    df = loader.load_data()
    df = loader.add_context(scrape_limit=5)
    print("\nSample Data with Context:")
    print(df[['headline', 'context', 'is_sarcastic']].head())
