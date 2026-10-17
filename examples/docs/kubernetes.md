# Kubernetes

Kubernetes is an open-source system for automating deployment, scaling and management of containerised applications. The control plane runs the API server, the scheduler and controllers that reconcile the desired state stored in etcd.

A pod is the smallest deployable unit and wraps one or more containers that share a network namespace. Deployments manage ReplicaSets, which keep a fixed number of identical pods running. The Horizontal Pod Autoscaler adjusts the replica count based on CPU usage or custom metrics.

Services give a stable virtual IP and DNS name to a changing set of pods, and an Ingress routes external HTTP traffic to services.
