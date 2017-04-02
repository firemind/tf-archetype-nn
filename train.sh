EXPORT_TO=/archetype-data/tf-model/0000000001
rm -r  $EXPORT_TO

#cd /serving/tensorflow_serving
#bazel build --config=opt //tensorflow_serving/firemind:archetype_export
/serving/bazel-bin/tensorflow_serving/firemind/archetype_export --training_iteration=2000 --model_version=1 $EXPORT_TO
