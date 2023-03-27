"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import json

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
import yaml
from yaml.loader import SafeLoader

from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline import Pipeline

cache_config = CacheConfig(enable_caching=True, expire_after="30d")

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def json_file_params(parent_folder, file):
    file_path = os.path.join(BASE_DIR, parent_folder, file) if file else None
    
    if file_path:
        with open(file_path, 'r') as f:
            params_dict = json.loads(f.read())

        return params_dict
    else:
        return None


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    training_image=None,
    preprocessing_script=None,
    training_script=None,
    hyperparameters=None,
    metric_definitions=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge"
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # validation of optional parameters
    entry_point = os.path.join(BASE_DIR, "training", training_script) if training_script else None
    hp_dict = json_file_params("hyperparameters", hyperparameters)
    md_dict = json_file_params("metric_definitions", metric_definitions)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
    )

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=pipeline_session,
        role=role
    )
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocessing", preprocessing_script),
        arguments=["--input-data", input_data],
    )
    step_process = ProcessingStep(
        name="PreprocessAbaloneData",
        step_args=step_args,
        cache_config=cache_config
    )

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/AbaloneTrain"
    xgb_train = Estimator(
        image_uri=training_image,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=pipeline_session,
        role=role,
        entry_point=entry_point,
        use_spot_instances=False,
        metric_definitions=md_dict,
        keep_alive_period_in_seconds=600,
    )
    
    xgb_train.set_hyperparameters(**hp_dict)
        
    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    step_train = TrainingStep(
        name="TrainAbaloneModel",
        step_args=step_args,
        cache_config=cache_config
    )

    # register model step that will be conditionally executed
    # model_metrics = ModelMetrics(
    #     model_statistics=MetricsSource(
    #         s3_uri="{}/evaluation.json".format(
    #             step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    #         ),
    #         content_type="application/json"
    #     )
    # )
    # model = Model(
    #     image_uri=training_image,
    #     model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    #     sagemaker_session=pipeline_session,
    #     role=role,
    # )
    # step_args = model.register(
    #     content_types=["text/csv"],
    #     response_types=["text/csv"],
    #     inference_instances=["ml.t2.medium", "ml.m5.large"],
    #     transform_instances=["ml.m5.large"],
    #     model_package_group_name=model_package_group_name,
    #     approval_status=model_approval_status,
    #     model_metrics=model_metrics,
    # )
    # step_register = ModelStep(
    #     name="RegisterAbaloneModel",
    #     step_args=step_args,
    # )

    # condition step for evaluating model quality and branching execution
    # cond_lte = ConditionLessThanOrEqualTo(
    #     left=JsonGet(
    #         step_name=step_eval.name,
    #         property_file=evaluation_report,
    #         json_path="regression_metrics.mse.value"
    #     ),
    #     right=6.0,
    # )
    # step_cond = ConditionStep(
    #     name="CheckMSEAbaloneEvaluation",
    #     conditions=[cond_lte],
    #     if_steps=[step_register],
    #     else_steps=[],
    # )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train],
        sagemaker_session=pipeline_session,
    )
    return pipeline
