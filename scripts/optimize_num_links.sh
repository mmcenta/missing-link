sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 7
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec7.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 8
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec8.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 9
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec9.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 10
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec10.txt