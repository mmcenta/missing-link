sudo python3 preprocess_text.py --representation_size 64 --use_tfidf --use_doc2vec
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 64 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > base64.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 64 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/doc2vec.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec64.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 64 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/reduced_tfidf.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf64.txt

sudo python3 preprocess_text.py --representation_size 128 --use_tfidf --use_doc2vec
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 128 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > base128.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 128 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/doc2vec.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec128.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 128 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/reduced_tfidf.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf128.txt

sudo python3 preprocess_text.py --representation_size 256 --use_tfidf --use_doc2vec
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 256 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > base256.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 256 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/doc2vec.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec256.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 256 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/reduced_tfidf.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf256.txt

sudo python3 preprocess_text.py --representation_size 512 --use_tfidf --use_doc2vec
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > base512.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/doc2vec.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec512.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/reduced_tfidf.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf512.txt

sudo python3 preprocess_text.py --representation_size 1024 --use_tfidf --use_doc2vec
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 1024 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > base1024.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 1024 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/doc2vec.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec1024.txt
deepwalk --format adjlist --input ./data/train_adjlist.txt --number-walks 10 --representation-size 1024 --walk-length 40 --window-size 2 --workers 16 --pretrained ./data/node_information/reduced_tfidf.embeddings --output ./data/node_information/deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf1024.txt