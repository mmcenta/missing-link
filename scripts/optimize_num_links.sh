deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf1.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 2
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf2.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 3
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf3.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 4
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf4.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 5
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf5.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 6
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf6.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 7
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf7.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 8
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf8.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 9
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf9.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --tfidf_file ./data/node_information/tfidf_embeddings.pickle --k 10
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > tfidf10.txt