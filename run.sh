### Workflow for books
docker-compose run app python3 -m src.parse.parse_json data/raw/meta_Books.json.gz data/interim/books.csv

docker-compose run app python3 -m src.prep.prep_node_relationship data/interim/books.csv data/interim/books_relationships.csv
python -m src.prep.prep_meta data/books.csv data/books_meta.csv

docker-compose run app python3 -m src.prep.prep_edges data/interim/books_relationships.csv data/interim/books_edges.csv
docker-compose run app python3 -m src.prep.train_val_split data/interim/books_edges.csv 0.33

docker-compose run app python3 -m src.prep.prep_graph_samples data/interim/books_edges_train.edgelist data/processed/books_sequences.npy books

# Slow and requires a lot of ram
python -m src.ml.train_node2vec_embeddings data/books_edges_train.edgelist data/books_embeddings.kv

# Works fine with multiprocess
python -m src.ml.train_gensim_embedding data/books_sequences_sample.npy 8

# PyTorch
# For dev testing
python -m src.ml.train_torch_embedding data/books_sequences_sample.npy data/books_edges_val_samp.csv data/books_edges_train_samp.csv 32 4
# For training
docker-compose run app python3 -m src.ml.train_torch_embedding data/processed/books_sequences.npy data/processed/books_edges_val.csv data/processed/books_edges_val.csv 128 10  # Best params?

# ==========================================================================================================================================
### Workflow for electronics
docker-compose run app python3 -m src.parse.parse_json data/raw/meta_Electronics.json.gz data/interim/electronics.csv

docker-compose run app python3 -m src.prep.prep_node_relationship data/interim/electronics.csv data/interim/electronics_relationships.csv
python -m src.prep.prep_meta data/electronics.csv data/electronics_meta.csv

docker-compose run app python3 -m src.prep.prep_edges data/interim/electronics_relationships.csv data/interim/electronics_edges.csv
docker-compose run app python3 -m src.prep.train_val_split data/interim/electronics_edges.csv 0.33

docker-compose run app python3 -m src.prep.prep_graph_samples data/interim/electronics_edges_train.edgelist data/processed/electronics_sequences.npy electronics

# Slow and requires a lot of ram
python -m src.ml.train_node2vec_embeddings data/electronics_edges_train.edgelist data/electronics_embeddings.kv

# Works fine with multiprocess
python -m src.ml.train_gensim_embedding data/electronics_sequences_sample.npy 6

# PyTorch
# For dev testing
python -m src.ml.train_torch_embedding data/electronics_sequences_samp.npy data/electronics_edges_val_samp.csv data/electronics_edges_train_samp.csv 32 4
python -m src.ml.train_torch_embedding_with_meta data/electronics_sequences_samp.npy data/electronics_edges_val_samp.csv data/electronics_meta.csv data/electronics_edges_train_samp.csv 32 4
# For training
docker-compose run app python3 -m src.ml.train_torch_embedding data/processed/electronics_sequences.npy data/processed/electronics_edges_val.csv  data/processed/electronics_edges_val.csv 128 4  # Best params?
python -m src.ml.train_torch_embedding_with_meta data/electronics_sequences.npy data/electronics_edges_val.csv data/electronics_meta.csv data/electronics_edges_val_samp.csv 128 10  # Best params?

# MF Dev
python -m src.ml.train_torch_mf data/electronics_sequences_samp.npy data/electronics_edges_val_samp.csv data/electronics_edges_val_samp.csv 32 4
python -m src.ml.train_torch_mf data/electronics_sequences.npy data/electronics_edges_val.csv  data/electronics_edges_val_samp.csv 128 8  # Best params?
python -m src.ml.train_torch_mf_bias data/electronics_sequences.npy data/electronics_edges_val.csv  data/electronics_edges_val_samp.csv 128 8  # Best params?

# Edges model
python -m src.ml.train_torch_mf_edges data/electronics_edges_train_samp.csv data/electronics_edges_val_samp.csv data/electronics_edges_val_samp.csv 32 4

# ==========================================================================================================================================
### Running for results
python -m src.ml.train_gensim_embedding data/electronics_sequences.npy 8
python -m src.ml.train_torch_embedding data/electronics_sequences.npy data/electronics_edges_val.csv  data/electronics_edges_val_samp.csv 128 8  # Best params?
python -m src.ml.train_torch_embedding_with_meta data/electronics_sequences.npy data/electronics_edges_val.csv data/electronics_meta.csv data/electronics_edges_val_samp.csv 128 8  # Best params?
python -m src.ml.train_torch_mf data/electronics_sequences.npy data/electronics_edges_val.csv  data/electronics_edges_val_samp.csv 128 8  # Best params?
python -m src.ml.train_torch_mf_bias data/electronics_sequences.npy data/electronics_edges_val.csv  data/electronics_edges_val_samp.csv 128 4  # Best params?
python -m src.ml.train_torch_mf_edges data/electronics_edges_train.csv data/electronics_edges_val.csv data/electronics_edges_val_samp.csv 128 8
python -m src.ml.train_torch_mf_bias_edges data/electronics_edges_train.csv data/electronics_edges_val.csv data/electronics_edges_val_samp.csv 128 4
python -m src.ml.train_torch_mf_continuous_edges data/electronics_edges_train.csv data/electronics_edges_val.csv data/electronics_edges_val_samp.csv 128 4
python -m src.ml.train_torch_mf_bias_continuous_edges data/electronics_edges_train.csv data/electronics_edges_val.csv data/electronics_edges_val_samp.csv 128 8


python -m src.ml.train_gensim_embedding data/books_sequences.npy 8
python -m src.ml.train_torch_embedding data/books_sequences.npy data/books_edges_val.csv  data/books_edges_val_samp.csv 128 8  # Best params?
python -m src.ml.train_torch_mf_bias data/books_sequences.npy data/books_edges_val.csv  data/books_edges_val_samp.csv 128 8  # Best params?
python -m src.ml.train_torch_mf_bias_edges data/books_edges_train.csv data/books_edges_val.csv data/books_edges_val_samp.csv 128 8
python -m src.ml.train_torch_mf_bias_continuous_edges data/books_edges_train.csv data/books_edges_val.csv data/books_edges_val_samp.csv 128 8


