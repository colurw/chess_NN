from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import JobService
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.ai.ml.entities import ResourceConfiguration
import os
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# join workspace
subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']  # ends with '2ba'
resource_group = "colurwin-rg" 
workspace = "chess3" 

ml_client = MLClient(AzureCliCredential(), subscription_id, resource_group, workspace)

# submit job
job = command(code='.',
              command='python _9a_pgn_trainer.py',
              environment='AzureML-tensorflow-2.12-cuda11@latest',
              #compute='gpu-cluster',
              display_name='chess_wholegame')

job.resources = ResourceConfiguration(instance_type='Standard_NC6s_v3',  # 112GB, V100 16GB, 50p/hour
                                      instance_count=2)                  # serverless compute 

returned_job = ml_client.create_or_update(job)

# save model
model = Model(path="azureml://jobs/{}/outputs/artifacts/paths/model/".format(returned_job.name),
              name="whole_game_4",
              description="model created from run",
              type=AssetTypes.MLFLOW_MODEL)

ml_client.models.create_or_update(model)