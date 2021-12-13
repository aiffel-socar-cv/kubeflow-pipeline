#titanic_kfp_pipeline.ipynb
#Copyright 2020 Google LLC. 
#This software is provided as-is, without warranty or representation for any use or purpose. 
#Your use of it is subject to your agreements with Google.
#Author: whjang@google.com
#!pip3 install -U kfp
import kfp
import kfp.components as comp
from kfp import dsl
from kfp import compiler
from kfp.components import func_to_container_op
import time
import datetime

PIPELINE_HOST = “55b5c3378a14c1c1-dot-us-west1.pipelines.googleusercontent.com”
WORK_BUCKET = “gs://aiplatformdemo-kubeflowpipelines-default”
EXPERIMENT_NAME = “Titanic Draft Experiment”

# Function for determine deployment
@func_to_container_op
def check_and_deploy_op(ACC_CSV_GCS_URI) -> str:
    import sys, subprocess
    subprocess.run([sys.executable, ‘-m’, ‘pip’, ‘install’, ‘pandas’])
    subprocess.run([sys.executable, ‘-m’, ‘pip’, ‘install’, ‘gcsfs’])
    import pandas as pd
    acc_df = pd.read_csv(ACC_CSV_GCS_URI)
    return acc_df[“deploy”].item()

@func_to_container_op
def finish_deploy_op(ACC_CSV_GCS_URI):
    import sys, subprocess
    subprocess.run([sys.executable, ‘-m’, ‘pip’, ‘install’, ‘pandas’])
    subprocess.run([sys.executable, ‘-m’, ‘pip’, ‘install’, ‘gcsfs’])
    import pandas as pd
    acc_df = pd.read_csv(ACC_CSV_GCS_URI)
    acc_df[“deploy”] = “done”
    acc_df.to_csv(ACC_CSV_GCS_URI)
    print(“Successfully new model was deployed”)

@dsl.pipeline(
    name=”titanic-kubeflow-pipeline-demo”,
    description = “Titanic Kubeflow Pipelines demo embrassing AI Platform in Google Cloud”
)

def titanic_pipeline(
    PROJECT_ID,
    WORK_BUCKET,
    RAW_CSV_GCS_URI,
    PREPROC_CSV_GCS_URI,
    ACC_CSV_GCS_URI,
    MODEL_PKL_GCS_URI,
    MIN_ACC_PROGRESS,
    STAGE_GCS_FOLDER,
    TRAIN_ON_CLOUD,
    AIPJOB_TRAINER_GCS_PATH,
    AIPJOB_OUTPUT_GCS_PATH
):
IMAGE_PREFIX = “whjang-titanic”
PREPROC_DIR = “preprocess”
TRAIN_DIR = “train”
MODEL_DIR = “model”

preprocess = dsl.ContainerOp(
    name = “Preprocess raw data and generate new one”,
    image = “gcr.io/” + str(PROJECT_ID) + “/” + IMAGE_PREFIX + “-” + PREPROC_DIR + “:latest”,
    arguments = [
    “--raw_csv_gcs_uri”, RAW_CSV_GCS_URI,
    “--preproc_csv_gcs_uri”, PREPROC_CSV_GCS_URI
    ]
    )
    train_args = [
    “--preproc_csv_gcs_uri”, str(PREPROC_CSV_GCS_URI),
    “--model_pkl_gcs_uri”, str(MODEL_PKL_GCS_URI),
    “--acc_csv_gcs_uri”, str(ACC_CSV_GCS_URI),
    “--min_acc_progress”, str(MIN_ACC_PROGRESS)
]

with dsl.Condition(TRAIN_ON_CLOUD == False) as check_condition1:
    train = dsl.ContainerOp(
    name = “Train”,
    image = “gcr.io/” + str(PROJECT_ID) + “/” + IMAGE_PREFIX + “-” + TRAIN_DIR + “:latest”,
    arguments = train_args,
    file_outputs={
    “mlpipeline-metrics” : “/mlpipeline-metrics.json”
    }
)

with dsl.Condition(TRAIN_ON_CLOUD == True) as check_condition2:
    aip_job_train_op = comp.load_component_from_url(“https://raw.githubusercontent.com/kubeflow/pipelines/1.0.0/components/gcp/ml_engine/train/component.yaml”)
    help(aip_job_train_op)
    aip_train = aip_job_train_op(
    project_id=PROJECT_ID, 
    python_module=”train.titanic_train”, 
    package_uris=json.dumps([str(AIPJOB_TRAINER_GCS_PATH)]), 
    region=”us-west1", 
    args=json.dumps(train_args),
    job_dir=AIPJOB_OUTPUT_GCS_PATH, 
    python_version=”3.7",
    runtime_version=”1.15", #cf. 2.1 
    master_image_uri=””, 
    worker_image_uri=””, 
    training_input=””, 
    job_id_prefix=””, 
    job_id=””,
    wait_interval=5
)

check_deploy = check_and_deploy_op(ACC_CSV_GCS_URI)
with dsl.Condition(check_deploy.output == “pending”):
    aip_model_deploy_op = comp.load_component_from_url(“https://raw.githubusercontent.com/kubeflow/pipelines/1.0.0/components/gcp/ml_engine/deploy/component.yaml”)
    help(aip_model_deploy_op)
    aip_model_deploy = aip_model_deploy_op(
    model_uri=str(WORK_BUCKET) + “/” + MODEL_DIR, 
    project_id=PROJECT_ID, 
    model_id=””, 
    version_id=””, 
    runtime_version=”1.15", #cf. 2.1 
    python_version=”3.7",
    version=””, 
    replace_existing_version=”False”, 
    set_default=”True”, 
    wait_interval=5
)

lastStep = finish_deploy_op(ACC_CSV_GCS_URI)

check_condition1.after(preprocess)
check_condition2.after(preprocess)
check_deploy.after(aip_train)
lastStep.after(aip_model_deploy)

train.execution_options.caching_strategy.max_cache_staleness = “P0D”
aip_train.execution_options.caching_strategy.max_cache_staleness = “P0D”
check_deploy.execution_options.caching_strategy.max_cache_staleness = “P0D”
aip_model_deploy.execution_options.caching_strategy.max_cache_staleness = “P0D”
lastStep.execution_options.caching_strategy.max_cache_staleness = “P0D”

args = {
    “PROJECT_ID” : “aiplatformdemo”,
    “WORK_BUCKET” : WORK_BUCKET,
    “RAW_CSV_GCS_URI” : WORK_BUCKET + “/rawdata/train.csv”,
    “PREPROC_CSV_GCS_URI” : WORK_BUCKET + “/preprocdata/processed_train.csv”,
    “ACC_CSV_GCS_URI” : WORK_BUCKET + “/latestacc/accuracy.csv”,
    “MODEL_PKL_GCS_URI” : WORK_BUCKET + “/model/model.pkl”,
    “MIN_ACC_PROGRESS” : 0.000001,
    “STAGE_GCS_FOLDER” : WORK_BUCKET + “/stage”,
    “TRAIN_ON_CLOUD” : False,
    “AIPJOB_TRAINER_GCS_PATH” : WORK_BUCKET + “/train/titanic_train.tar.gz”,
    “AIPJOB_OUTPUT_GCS_PATH” : WORK_BUCKET + “/train/output/”
}

client = kfp.Client(host=PIPELINE_HOST)
#pipeline_name = “titanic_pipelines.zip”
#compiler.Compiler().compile(titanic_pipeline, pipeline_name)
#try:
# pipeline = client.upload_pipeline(pipeline_package_path=pipeline_name, pipeline_name=pipeline_name)
# print(“uploaded:” + pipeline.id)
#except:
# print(“already exist”)
client.create_run_from_pipeline_func(
titanic_pipeline,
arguments=args,
experiment_name=EXPERIMENT_NAME
)