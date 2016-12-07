# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_library(
    name = "archetype_input_data",
    srcs = ["archetype_input_data.py"],
)

py_binary(
    name = "archetype_export",
    srcs = [
        "archetype_export.py",
    ],
    deps = [
        ":archetype_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "archetype_client",
    srcs = [
        "archetype_client.py",
    ],
    deps = [
        ":archetype_input_data",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "inception_export",
    srcs = [
        "inception_export.py",
    ],
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "inception_client",
    srcs = [
        "inception_client.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
