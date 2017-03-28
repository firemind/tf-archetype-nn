EXPORT_TO=/myproject/firemind/model/0000000001
rm -r  $EXPORT_TO
/serving/bazel-bin/tensorflow_serving/firemind/archetype_export --training_iteration=2000 --model_version=1 $EXPORT_TO
