
sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 11
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf11.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 12
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf12.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 13
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf13.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 14
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf14.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 15
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf15.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 16
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf16.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 17
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf17.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 18
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf18.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 19
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf19.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 20
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf20.txt