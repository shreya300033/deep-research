"""
Document processing utilities for various file formats
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from .pdf_parser import process_pdf_file, process_pdf_data


class DocumentProcessor:
    """Handles processing of various document formats"""
    
    @staticmethod
    def process_text_file(file_path: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Process a plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if title is None:
                title = Path(file_path).stem
            
            return {
                'id': f"text_{Path(file_path).stem}",
                'title': title,
                'content': content,
                'source': file_path,
                'type': 'text'
            }
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            return None
    
    @staticmethod
    def process_json_file(file_path: str) -> List[Dict[str, Any]]:
        """Process a JSON file containing documents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        doc = {
                            'id': item.get('id', f"json_{i}"),
                            'title': item.get('title', f"Document {i}"),
                            'content': item.get('content', str(item)),
                            'source': file_path,
                            'type': 'json'
                        }
                        documents.append(doc)
            elif isinstance(data, dict):
                doc = {
                    'id': data.get('id', 'json_document'),
                    'title': data.get('title', 'JSON Document'),
                    'content': data.get('content', str(data)),
                    'source': file_path,
                    'type': 'json'
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error processing JSON file {file_path}: {e}")
            return []
    
    @staticmethod
    def process_pdf_file(file_path: str) -> Dict[str, Any]:
        """Process a PDF file"""
        try:
            return process_pdf_file(file_path)
        except Exception as e:
            print(f"Error processing PDF file {file_path}: {e}")
            return None
    
    @staticmethod
    def process_pdf_data(pdf_data: bytes, filename: str) -> Dict[str, Any]:
        """Process PDF data from bytes"""
        try:
            return process_pdf_data(pdf_data, filename)
        except Exception as e:
            print(f"Error processing PDF data {filename}: {e}")
            return None
    
    @staticmethod
    def process_directory(directory_path: str, file_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Process all files in a directory"""
        if file_extensions is None:
            file_extensions = ['.txt', '.json', '.md', '.pdf']
        
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Directory {directory_path} does not exist")
            return documents
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                if file_path.suffix.lower() == '.txt':
                    doc = DocumentProcessor.process_text_file(str(file_path))
                    if doc:
                        documents.append(doc)
                elif file_path.suffix.lower() == '.json':
                    docs = DocumentProcessor.process_json_file(str(file_path))
                    documents.extend(docs)
                elif file_path.suffix.lower() == '.md':
                    doc = DocumentProcessor.process_text_file(str(file_path))
                    if doc:
                        documents.append(doc)
                elif file_path.suffix.lower() == '.pdf':
                    doc = DocumentProcessor.process_pdf_file(str(file_path))
                    if doc:
                        documents.append(doc)
        
        return documents
    
    @staticmethod
    def create_sample_documents() -> List[Dict[str, Any]]:
        """Create sample documents for testing"""
        sample_docs = [
            {
                'id': 'ai_basics',
                'title': 'Introduction to Artificial Intelligence',
                'content': '''
                Artificial Intelligence (AI) is a branch of computer science that aims to create 
                intelligent machines that can perform tasks that typically require human intelligence. 
                These tasks include learning, reasoning, problem-solving, perception, and language understanding.
                
                AI can be categorized into two main types: Narrow AI and General AI. Narrow AI is designed 
                to perform specific tasks, while General AI would have the ability to understand, learn, 
                and apply knowledge across different domains.
                
                Machine Learning is a subset of AI that focuses on algorithms that can learn from data. 
                Deep Learning, in turn, is a subset of machine learning that uses neural networks with 
                multiple layers to model and understand complex patterns in data.
                ''',
                'source': 'sample_data',
                'type': 'sample'
            },
            {
                'id': 'ml_algorithms',
                'title': 'Machine Learning Algorithms Overview',
                'content': '''
                Machine Learning algorithms can be broadly classified into three categories: 
                Supervised Learning, Unsupervised Learning, and Reinforcement Learning.
                
                Supervised Learning algorithms learn from labeled training data to make predictions 
                on new, unseen data. Common algorithms include Linear Regression, Decision Trees, 
                Random Forest, and Support Vector Machines.
                
                Unsupervised Learning algorithms find hidden patterns in data without labeled examples. 
                Examples include K-Means clustering, Hierarchical clustering, and Principal Component Analysis.
                
                Reinforcement Learning involves an agent learning to make decisions by taking actions 
                in an environment to maximize cumulative reward. This approach is used in game playing, 
                robotics, and autonomous systems.
                ''',
                'source': 'sample_data',
                'type': 'sample'
            },
            {
                'id': 'nlp_fundamentals',
                'title': 'Natural Language Processing Fundamentals',
                'content': '''
                Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
                between computers and humans through natural language. The ultimate objective of NLP 
                is to read, decipher, understand, and make sense of human language in a valuable way.
                
                Key NLP tasks include:
                - Text Classification: Categorizing text into predefined classes
                - Named Entity Recognition: Identifying and classifying named entities
                - Sentiment Analysis: Determining the emotional tone of text
                - Machine Translation: Translating text from one language to another
                - Question Answering: Automatically answering questions posed in natural language
                
                Modern NLP systems often use transformer-based models like BERT, GPT, and T5, 
                which have achieved state-of-the-art performance on many NLP tasks.
                ''',
                'source': 'sample_data',
                'type': 'sample'
            },
            {
                'id': 'computer_vision',
                'title': 'Computer Vision and Image Processing',
                'content': '''
                Computer Vision is a field of AI that trains computers to interpret and understand 
                the visual world. Using digital images from cameras and videos and deep learning models, 
                machines can accurately identify and classify objects and react to what they see.
                
                Common computer vision tasks include:
                - Image Classification: Identifying objects in images
                - Object Detection: Locating and classifying multiple objects in images
                - Image Segmentation: Partitioning images into meaningful regions
                - Face Recognition: Identifying or verifying individuals from facial images
                - Optical Character Recognition: Extracting text from images
                
                Convolutional Neural Networks (CNNs) are the primary architecture used in computer vision, 
                with models like ResNet, VGG, and EfficientNet achieving excellent performance on various tasks.
                ''',
                'source': 'sample_data',
                'type': 'sample'
            },
            {
                'id': 'ai_ethics',
                'title': 'Ethics in Artificial Intelligence',
                'content': '''
                As AI systems become more powerful and widespread, ethical considerations become increasingly 
                important. Key ethical issues in AI include:
                
                - Bias and Fairness: AI systems can perpetuate or amplify human biases present in training data
                - Privacy: AI systems often require large amounts of personal data
                - Transparency: Many AI systems operate as "black boxes" with unclear decision-making processes
                - Accountability: Determining responsibility when AI systems make harmful decisions
                - Job Displacement: The potential impact of AI automation on employment
                
                Responsible AI development involves considering these ethical implications throughout the 
                design and deployment process, implementing fairness measures, ensuring transparency, 
                and maintaining human oversight of AI systems.
                ''',
                'source': 'sample_data',
                'type': 'sample'
            },
            {
                'id': 'large_language_models',
                'title': 'Large Language Models (LLMs)',
                'content': '''
                Large Language Models (LLMs) are a type of artificial intelligence system that uses deep learning 
                techniques to process and generate human-like text. These models are trained on vast amounts of 
                text data from the internet, books, articles, and other sources.
                
                Key characteristics of LLMs:
                
                - Scale: LLMs contain billions or even trillions of parameters, making them extremely large
                - Training Data: They are trained on massive datasets containing text from diverse sources
                - Transformer Architecture: Most modern LLMs use the transformer neural network architecture
                - Generative Capabilities: They can generate coherent, contextually relevant text
                - Few-shot Learning: They can perform new tasks with minimal examples or instructions
                
                Popular examples of LLMs include:
                - GPT (Generative Pre-trained Transformer) series by OpenAI
                - BERT (Bidirectional Encoder Representations from Transformers) by Google
                - T5 (Text-to-Text Transfer Transformer) by Google
                - PaLM (Pathways Language Model) by Google
                - LLaMA (Large Language Model Meta AI) by Meta
                
                Applications of LLMs:
                - Text generation and completion
                - Question answering and chatbots
                - Language translation
                - Code generation and programming assistance
                - Content summarization
                - Creative writing and storytelling
                
                LLMs represent a significant advancement in natural language processing and have revolutionized 
                how we interact with AI systems for text-based tasks.
                ''',
                'source': 'sample_data',
                'type': 'sample'
            }
        ]
        
        return sample_docs
