# Concept Representation
Phase 3 of Lecture Mapping project

## Goal
- Group the keywords and phrases extracted by NP chunking NN into sets that describe the same concept
- Define a concept as this set of words and phrases

## Game Plan
- Define the 'neighborhood' of a keyword as: M sentences before the sentence containing the keyword + M sentences after
- Use Topic Modeling (Gensim Python library: LDA) on each highlighted keyword's neighborhood to get a word distribution that describes that neighborhood
- Use KL-divergence between these distributions to group together concept keywords that describe the same concept according to a threshold T

