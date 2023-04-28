# ML Experiments at scale Deployment

## Purpose

The purpose of this template is to deploy the base infrastructure to build models via sagemaker pipelines. In addition to this infrastructure, there is additional code setup that will allow the user to run experiments at scale by just providing the experiments configuration.

## Architecture

![exp-at-scale.png](images/exp-at-scale.png)

## Instructions

Part 1: Create initial Service Catalog Product

1. To create the Service Catalog product for this project, download the `create-ml-experiments-product.yaml` and upload it into your CloudFormation console: https://console.aws.amazon.com/cloudformation


2. Update the Parameters section:

    - Supply a unique name for the stack

        ![](images/stack-name-01.png)

    - Enter your Service Catalog portfolio id, which can be found in the __Outputs__ tab of your deployed portfolio stack or in the Service Catalog portfolio list: https://console.aws.amazon.com/servicecatalog/home?#/portfolios

        ![](images/portfolio-ID-02.png)

    - Update the Product Information. The product name and description are visible inside of SageMaker Studio. Other fields are visible to users that consume this directly through Service Catalog. 

    - Support information is not available inside of SageMaker Studio, but is available in the Service Catalog Dashboard.

    - Updating the source code repository information is only necessary if you forked this repo and modified it.

        ![](images/source-code-05.png)

3. Choose __Next__, __Next__ again, check the box acknowledging that the template will create IAM resources, and then choose __Create Stack__.

4. Your template should now be visible inside of SageMaker Studio.


Part 2: Deploy the Project inside of SageMaker Studio

1. Open SageMaker Studio and sign in to your user profile.

2. Choose the SageMaker __components and registries__ icon on the left, and choose the __Create project__ button.

3. The default view displays SageMaker templates. Switch to the __Organization__ templates tab to see custom project templates.

4. The template you created will be displayed in the template list. (If you do not see it yet, make sure the correct execution role is added to the product and the __sagemaker:studio-visibility__ tag with a value of __true__ is added to the Service Catalog product).

5. Choose the template and click Select the correct project template.

    ![](images/select-project.png)

6. Fill out the required fields for this project.

    - __Name:__ A unique name for the project deployment.

    - __Description:__ Project description for this deployment.

    - __CodeTypeComputeBuild:__ The type of instance used for the CodeBuild project that runs the code that launches the pipelines. By default is the smallest type of instance, but if needed it can be changed (see more options [here](https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-compute-types.html)).

7. Choose __Create Project__.

    ![](images/create-project.png)

8. After a few minutes, your example project should be deployed and ready to use.