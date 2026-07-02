import json
import boto3
import os
import uuid

mediaconvert = boto3.client('mediaconvert')

def lambda_handler(event, context):
    assetID = str(uuid.uuid4())
    srcbucket = event['Records'][0]['s3']['bucket']['name']
    srcobject = event['Records'][0]['s3']['object']['key']
    assetID = srcobject.rsplit('.', 1)[0]
    sources3 = 's3://' + srcbucket + '/' + srcobject
    dests3 = 's3://' + os.environ['destbucket'] + '/'
    region = os.environ['AWS_DEFAULT_REGION']
    mediaConvertRole = os.environ['MediaConvertRole']

    jobMetadata = {'assetID': assetID, 'copyright': 'safestart.com 2022'}
    
    try:
        # Job settings are in the lambda zip file in the current working directory
        with open('job.json') as json_data:
            jobSettings = json.load(json_data)

        # Update the job settings with the source video from the S3 event and destination paths for converted videos
        jobSettings['Inputs'][0]['FileInput'] = sources3
        
        # Sources3Basename for video
        S3KeyHLS = assetID + '/' + 'playlist'
        jobSettings['OutputGroups'][0]['OutputGroupSettings']['HlsGroupSettings']['Destination'] = dests3 + S3KeyHLS

        # Sources3Basename for thumbnail
        S3KeyThumbnails =  assetID + '/thumb/thumbnail'
        jobSettings['OutputGroups'][1]['OutputGroupSettings']['FileGroupSettings']['Destination'] = dests3 + S3KeyThumbnails     

        print('JobSettings: \n'+json.dumps(jobSettings))

        # Convert the video using AWS Elemental MediaConvert
        job = mediaconvert.create_job(Role=mediaConvertRole, UserMetadata=jobMetadata, Settings=jobSettings)
        job_execution_id = job['Job']['Id']

        print('Completed Job id: '+job_execution_id)
        return {'SUCCEEDED': f'{job_execution_id}'}
        
    except Exception as e:
        print ('Exception: %s' % e)
        statusCode = 500
        raise