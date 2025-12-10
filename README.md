# Medical Record Processing and Feature Extraction System

## Overview

This codebase is designed to extract structured medical features from unstructured medical records using Large Language Models (LLMs). The system processes patient data from the Research Patient Data Registry (RPDR) and extracts various clinical features like smoking status, cancer history, medication use, and more.

## Core Architecture

### 1. Data Pipeline (`make_db.py`)
- **Purpose**: Processes raw RPDR data files and creates a SQLite database
- **Input**: Pipe-delimited text files from RPDR (demographics, diagnoses, medications, visits, etc.)
- **Output**: SQLite database with structured tables for each data type
- **Key Tables**: `vis` (visits), `dia` (diagnoses), `med` (medications), `dem` (demographics)

### 2. Feature Definition System (`pt_features.py`)
- **Purpose**: Defines all medical features to be extracted from patient records
- **Architecture**: Uses metaclasses to register feature classes automatically
- **Feature Types**:
  - `PtFeatureBase`: Basic features (smoking, alcohol, cancer, etc.)
  - `PtDateFeatureBase`: Date-based features (diagnosis dates, treatment dates)
- **Key Components**:
  - `query`: LLM prompt for feature extraction
  - `keywords`: List of keywords to identify relevant text chunks
  - `options`: Multiple choice options for the LLM
  - `pooling_fn`: Function to aggregate predictions across multiple text chunks

### 3. Text Processing (`utils.py`)
- **Purpose**: Utilities for text chunking and keyword detection
- **Key Functions**:
  - `chunk_text()`: Splits long medical reports into manageable chunks
  - `has_keyword()`: Uses FlashText to efficiently detect keywords in text

### 4. LLM Integration (`models.py`)
- **Purpose**: Interfaces with LLMs for feature extraction
- **Model**: Uses vLLM with Llama-4-Scout-17B for inference
- **Key Methods**:
  - `format_chunk_qs()`: Formats text chunks into LLM prompts
  - `predict_single_with_logit_trick()`: Uses logprobs for more reliable predictions

### 5. Main Processing Pipeline (`process_notes.py`)
- **Purpose**: Orchestrates the entire feature extraction process
- **Workflow**:
  1. Query database for records with null feature values
  2. Chunk medical reports into smaller pieces
  3. Filter chunks containing relevant keywords
  4. Send keyword-containing chunks to LLM for prediction
  5. Aggregate predictions across chunks per report
  6. Save results back to database

### 6. Validation Dataset Creation (`make_val_ds.py`)
- **Purpose**: Creates validation datasets for manual review
- **Process**:
  1. Samples random medical reports from database
  2. Identifies reports containing feature-specific keywords
  3. Saves relevant reports with keyword annotations to CSV files
  4. Creates metadata files documenting the extraction criteria

## Key Data Flow

```
RPDR Raw Files → make_db.py → SQLite Database
                                    ↓
Feature Definitions (pt_features.py) → process_notes.py → LLM Predictions
                                    ↓
                            Updated Database + Validation Datasets
```

## Feature Categories

The system extracts features across several medical domains:

- **Lifestyle**: Smoking status, alcohol use, military service
- **Medical History**: Cancer diagnoses, transplant history, infections
- **Medications**: Antibiotic use, hormone replacement therapy
- **Comorbidities**: Diabetes, autoimmune diseases, skin conditions
- **Demographics**: Age, sex, race, ethnicity, insurance status

## Configuration

- **Database**: SQLite (`stores/rpdr.db`)
- **LLM**: Llama-4-Scout-17B via vLLM server
- **Chunk Size**: 200 words with 20-word overlap
- **Batch Processing**: Configurable batch sizes for efficiency

## Usage Patterns

1. **Initial Setup**: Run `make_db.py` to create database from RPDR files
2. **Feature Extraction**: Run `process_notes.py` to extract features using LLMs
3. **Validation**: Run `make_val_ds.py` to create datasets for manual review
4. **Evaluation**: Use `compare_to_gt.py` to compare predictions against ground truth

## Key Design Principles

- **Modularity**: Features are defined as separate classes for easy addition/modification
- **Efficiency**: Keyword filtering reduces unnecessary LLM calls
- **Reliability**: Logprob-based prediction ensures consistent outputs
- **Scalability**: Batch processing and database storage handle large datasets
- **Validation**: Built-in validation dataset creation for quality assurance
This system enables automated extraction of structured medical features from unstructured clinical text, supporting research and clinical decision-making applications. 
