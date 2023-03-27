import os
import json
from pipelines.abalone.pipeline import get_pipeline

json_file_path = "config.json"

print("###### Loading experiments configuration file...")

with open(json_file_path, 'r') as j:
    experiments = json.loads(j.read())

tags = [{"Key": "sagemaker:project-name", "Value": os.getenv("SAGEMAKER_PROJECT_NAME")},
        {"Key": "sagemaker:project-id", "Value": os.getenv("SAGEMAKER_PROJECT_ID")}]

AWS_REGION = os.getenv("AWS_REGION")
SAGEMAKER_PIPELINE_ROLE_ARN = os.getenv("SAGEMAKER_PIPELINE_ROLE_ARN")
ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET")
SAGEMAKER_PROJECT_NAME_ID = os.getenv("SAGEMAKER_PROJECT_NAME_ID")


def main():

    for experiment in experiments:
        pipeline = get_pipeline(
            region=AWS_REGION,
            role=SAGEMAKER_PIPELINE_ROLE_ARN,
            default_bucket=ARTIFACT_BUCKET,
            model_package_group_name=SAGEMAKER_PROJECT_NAME_ID,
            pipeline_name=SAGEMAKER_PROJECT_NAME_ID,
            base_job_prefix=SAGEMAKER_PROJECT_NAME_ID,
            training_image=experiment["TrainingImage"],
            preprocessing_script=experiment["ProcessingScript"],
            training_script=experiment["TrainingScript"],
            hyperparameters=experiment["Hyperparameters"],
            metric_definitions=experiment["MetricDefinitions"],
            training_instance_type=experiment["TrainingInstanceType"]
        )

        print("###### Creating/updating a SageMaker Pipeline(s) with the following definition:")
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        upsert_response = pipeline.upsert(role_arn=SAGEMAKER_PIPELINE_ROLE_ARN, tags=tags)

        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

        execution = pipeline.start()
        print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")


if __name__ == "__main__":
    main()
