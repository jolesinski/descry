## Test data

Test code requires an example model and scene clouds.
The model has to be a 360 color scanned cloud with target object
and the scene is a color cloud containg the model object,
taken from a Kinect-like projective device.
To run tests over selected data, put them here, named:
* test_model.pcd
* test_scene.pcd

Performance tests require full model database with Willow-like structure.
To run the test code, you need to generate a willow.yaml with database paths, with:
>> ./emit_willow_conf.sh PATH_TO_DATABASE > descry/test/data/willow.yaml

For more information on the Willow database, please see:
https://repo.acin.tuwien.ac.at/tmp/permanent/dataset_index.php
