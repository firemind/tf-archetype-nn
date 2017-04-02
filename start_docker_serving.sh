#sudo docker run -it -p 9000:9000 -v $(pwd)/:/myproject  firemind/tf-serving
sudo docker run -it -p 9000:9000 -v /mnt/fileshare/archetype-data/:/archetype-data/ -v $(pwd)/:/myproject  firemind/tf-archetypes
