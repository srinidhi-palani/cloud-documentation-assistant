import json
import traceback
from ingestion.ingest import run_ingestion


def lambda_handler(event, context):
    """
    Lambda handler for ingestion pipeline
    Triggered by EventBridge cron schedule
    """
    print("Ingestion Lambda triggered")
    print(f"Event: {json.dumps(event)}")

    try:
        # Run the full ingestion pipeline
        run_ingestion()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Ingestion pipeline completed successfully",
                "status": "success"
            })
        }

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()

        print(f"Ingestion pipeline failed: {error_msg}")
        print(f"Traceback: {traceback_str}")

        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "Ingestion pipeline failed",
                "error": error_msg,
                "status": "failed"
            })
        }