sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 15
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
 sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec15.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 16
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec16.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 17
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec17.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 18
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec18.txt

sudo python3 generate_adjlist.py --input_file ./data/train.txt --output_file ./data/tmp_adjlist.txt --embeddings_file ./data/node_information/doc2vec.embeddings --num_potential_links 19
deepwalk --format adjlist --input ./data/tmp_adjlist.txt --number-walks 10 --representation-size 512 --walk-length 40 --window-size 2 --workers 16 --output ./data/node_information/tmp_deepwalk.embeddings
sudo python3 transform_dataset.py --input_file ./data/train.txt --output_name tmp  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 transform_dataset.py --input_file ./data/val.txt --output_name tmp_val  --graph_embeddings_file ./data/node_information/tmp_deepwalk.embeddings --concatenate
sudo python3 train_logreg.py --train_name tmp --val_name tmp_val > doc2vec19.txt
