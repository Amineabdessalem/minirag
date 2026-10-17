# Serverless computing

Serverless platforms such as AWS Lambda run functions in response to events and bill per invocation and duration. The provider handles provisioning, scaling to zero and patching.

A cold start happens when a new execution environment must be created before the function can run, which adds latency. Provisioned concurrency keeps environments warm. Functions are stateless, so state lives in managed services like DynamoDB or object storage.
