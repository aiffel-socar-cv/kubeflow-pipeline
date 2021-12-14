from typing import NamedTuple
import kfp
import kfp.dsl as dsl
import kfp.gcp as gcp
from kfp import components as comp
from typing import NamedTuple


def check_cnt(
    bucket_name: str, data_type: str
) -> NamedTuple("output", [("count", int), ("type", str)]):
    """
        count the number of newly added images
    """
    from google.cloud import storage

    client = storage.Client()
    blob = client.list_blobs("images-original", prefix="originals")
    cnt = -1  # root directory 제외
    for b in blob:
        cnt += 1


def retrain_op():
    return dsl.ContainerOp(
        name="retrain",
        image="us-west1-docker.pkg.dev/viai/retrain:v1.0",
        arguments=[
            "/opt/retrain.py",
            "--data-dir",
            data_dir,
            "--torch-export-dir",
            model_export_dir,
        ],
    )


@dsl.pipeline(name="retrain", description="retrain demo")
def retrain_pipeline(
    name="retrain",
    training_image="gcr.io/aiffel-gn-3/retrain:latest",
    training_namespace="kubeflow",
    model_export_dir="gs://model-cpt/",
):
    comp.func_to_container_op(check_cnt)
    check_cnt = check_op()
    with dsl.Condition(check_cnt.output > 100):
        retrain = retrain_op()


steps = [retrain]
for step in steps:
    if platform == "GCP":
        step.apply(gcp.use_gcp_secret("user-gcp-sa"))
if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(retrain_pipeline, __file__ + ".tar.gz")
