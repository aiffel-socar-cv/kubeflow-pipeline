import kfp
import kfp.components as comp
from kfp import dsl


@dsl.pipeline(name="viai-retrain", description="viai retrain pipeline")
def retrain_pipeline():
    check_files = dsl.ContainerOp(
        name="Check files",
        image="tseo/check_bucket:0.3",
        file_outputs={"file_num": "/file_nums.json"},
    )

    mv_files = dsl.ContainerOp(
        name="Move files",
        image="tseo/mv_files:0.6",
        arguments=["--json_file", check_files.outputs["file_num"]],
    )

    mv_files.after(check_files)


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(retrain_pipeline, __file__ + ".tar.gz")
