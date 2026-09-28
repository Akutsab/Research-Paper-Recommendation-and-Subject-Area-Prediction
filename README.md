Subject Prediction Pipeline:-

                ARXIV DATASET
                     │
                     ▼
              Data Cleaning
                     │
                     ▼
          Abstract + Subject Labels
                     │
          ┌──────────┴──────────┐
          │                     │
       Abstracts              Terms
          │                     │
          │              StringLookup
          │                     │
          │                     ▼
          │                Multi-hot
          │                labels
          │                     │
          ▼                     │
    TextVectorization           │
          │                     │
       TF-IDF                   │
          │                     │
          └──────────┬──────────┘
                     ▼
                    MLP
                     │
                     ▼
              Sigmoid outputs
                     │
                     ▼
            Subject predictions

Whole Project pipeline:-
                  RESEARCH PAPER SYSTEM
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       SUBJECT PREDICTION          PAPER RECOMMENDATION
              │                           │
          Abstract                       Title
              │                           │
              ▼                           ▼
       TextVectorization              MiniLM
          (tf-tdf)                  semantic embedding representation
word/statistical representation
              │                           │
              ▼                           ▼
            TF-IDF                   Embedding
              │                           │
              ▼                           ▼
             MLP                 Cosine Similarity
              │                           │
              ▼                           ▼
       Subject Categories             Top 5 Papers
