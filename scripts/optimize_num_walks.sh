deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 30 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > nw30.txt

deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 50 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > nw50.txt

deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 60 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > nw60.txt

deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 70 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > nw70.txt
